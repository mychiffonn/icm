"""Internal Coherence Maximization (ICM) Search Module."""

import logging
import math
import random
from dataclasses import dataclass
from typing import Optional

import openai
from tqdm import tqdm

from src import BASE_MODEL, SEED
from src.utils import (
    Example,
    construct_few_shot_prompt,
    extract_label_probs,
    get_hyperbolic_client,
)

logger = logging.getLogger(__name__)
random.seed(SEED)


@dataclass
class ICMArgs:
    """Arguments for ICM search.

    sample_size: Number of examples to sample for initial dataset (K in paper).
    initial_temperature: Starting temperature for simulated annealing (T_0).
    final_temperature: Minimum temperature (T_min).
    cooling_rate: Cooling rate parameter (β). Lower values = slower cooling = more exploration.
    max_iterations: Maximum number of search iterations (N).
    early_stop_patience: Stop if no improvement compared to best utility score, for this many iterations.
    early_stop_threshold: Stop if temperature drops below this value.
    """

    sample_size: int = 8
    initial_temperature: float = 10.0
    final_temperature: float = 0.01
    cooling_rate: float = 0.99
    max_iterations: int = 100
    early_stop_patience: int = 15
    early_stop_threshold: float = 0.001


@dataclass
class ICMState:
    """State of ICM search at any iteration.

    dataset: Current labeled dataset D.
    temperature: Current temperature T.
    iteration: Current iteration number.
    utility_score: Current mutual predictability score
    """

    dataset: list[Example]
    temperature: float
    iteration: int
    utility_score: float


@dataclass
class ICMResult:
    """Results from ICM search.

    final_dataset: Final labeled dataset after search.
    final_utility: Final utility score.
    utility_history: Utility score at each iteration (for convergence analysis).
    acceptance_history: Boolean indicating acceptance/rejection at each iteration.
    best_dataset: Best dataset seen during search (highest utility).
    best_utility: Best utility score seen during search.
    """

    final_dataset: list[Example]
    final_utility: float
    utility_history: list[float]
    acceptance_history: list[bool]
    best_dataset: list[Example]
    best_utility: float


class ICMSearch:
    """Internal Coherence Maximization search using simulated annealing."""

    def __init__(
        self,
        pool_data: list[Example],
        model: str = BASE_MODEL,
        args: Optional[ICMArgs] = None,
    ):
        """Initialize ICM search.

        Args:
            pool_data: Pool of examples to select from.
            model: Model name (base model).
            args: ICM hyperparameters. If None, uses defaults.
        """
        self.client = get_hyperbolic_client()
        self.model = model
        self.pool_data = pool_data
        self.args = args if args is not None else ICMArgs()

        # Initialstate
        self.current_dataset = self._initialize_dataset()
        self.temperature = self.args.initial_temperature
        self.iteration = 0

        # Track history
        self.utility_history: list[float] = []
        self.acceptance_history: list[bool] = []

        # Track best seen
        self.best_dataset = self.current_dataset.copy()
        self.best_utility = -float("inf")
        self.iterations_since_improvement = 0

    def _initialize_dataset(self) -> list[Example]:
        """Initialize dataset D by sampling K examples with random labels.

        Returns:
            List of K Example instances with random binary labels.
        """
        # Sample K examples from pool
        sampled = random.sample(self.pool_data, self.args.sample_size)

        # Assign random labels (deep copy to avoid modifying originals)
        dataset = []
        for ex in sampled:
            new_ex = Example(
                question=ex.question,
                choice=ex.choice,
                label=random.randint(0, 1),
                consistency_id=ex.consistency_id,
            )
            dataset.append(new_ex)

        return dataset

    def _compute_utility(self, dataset: list[Example]) -> float:
        """Compute mutual predictability (without consistency fix).

        Args:
            dataset: Labeled dataset D.

        Returns:
            Utility score (sum of log probabilities).
        """
        total_log_prob = 0.0

        for i, example in enumerate(dataset):
            # Context: D \ {(x_i, y_i)}
            context = dataset[:i] + dataset[i + 1 :]

            # Construct prompt
            prompt = construct_few_shot_prompt(
                examples=context, query_example=example, include_query_label=False
            )

            # Get normalized probabilities
            prob_0, prob_1 = extract_label_probs(self.client, prompt, self.model)

            # Get P(y_i | x_i, context) for the actual label
            prob = prob_0 if example.label == 0 else prob_1

            # Compute log probability
            total_log_prob += math.log(prob)

        return total_log_prob

    def _update_temperature(self) -> float:
        """Compute new temperature using cooling schedule.

        T = max(T_min, T_0 / (1 + beta * log(n+1)))

        Returns:
            New temperature.
        """
        n = self.iteration
        t_0 = self.args.initial_temperature
        t_min = self.args.final_temperature
        beta = self.args.cooling_rate

        # n+1 because n starts at 0
        t = t_0 / (1 + beta * math.log(n + 1))
        return max(t_min, t)

    def _acceptance_probability(self, delta_utility: float) -> float:
        """Compute acceptance probability using Metropolis criterion.

        Accept if Delta_U > 0 (improvement) OR
        Accept with probability exp(Delta_U/T) if Delta_U < 0 (worse)

        Args:
            delta_utility: Change in utility (U_new - U_old).

        Returns:
            Acceptance probability.
        """
        if delta_utility > 0:
            return 1.0
        else:
            return math.exp(delta_utility / self.temperature)

    def step(self) -> ICMState:
        """Execute one iteration of ICM search.

        Follows Algorithm 1 from the paper (without consistency fix):
        1. Sample example x_i from pool
        2. Assign best label: ŷ_i = argmax_y P(y|x,D_{x,y})
        3. Update D ← D ∪ {(x_i, y_i)}
        4. Accept/reject based on utility change

        Returns:
            Current state after this step.
        """
        # Sample example x_i from unlabeled pool
        sampled_example = random.choice(self.pool_data)

        # If this example is already in current dataset, remove it temporarily for argmax computation
        temp_dataset = [
            ex
            for ex in self.current_dataset
            if not (
                ex.question == sampled_example.question
                and ex.choice == sampled_example.choice
            )
        ]

        prompt = construct_few_shot_prompt(
            examples=temp_dataset,
            query_example=sampled_example,
            include_query_label=False,
        )

        # Assign best label: ŷ_i = argmax_y P(y|x, D\{(x,y)})
        # Get normalized probabilities for both labels
        prob_0, prob_1 = extract_label_probs(self.client, prompt, self.model)

        # Choose label with higher probability
        best_label = 0 if prob_0 > prob_1 else 1

        # Create new labeled example
        new_labeled_example = Example(
            question=sampled_example.question,
            choice=sampled_example.choice,
            label=best_label,
            consistency_id=sampled_example.consistency_id,
        )

        # Proposed dataset: D <- D ∪ {(x_i​,y^​i​)}
        proposed_dataset = temp_dataset + [new_labeled_example]

        # Compute new utility
        u_current = (
            self.utility_history[-1]
            if self.utility_history
            else self._compute_utility(self.current_dataset)
        )
        u_proposed = self._compute_utility(proposed_dataset)

        # Accept proposal based on delta_u and Metropolis criterion
        delta_u = u_proposed - u_current

        if delta_u > 0:
            self.current_dataset = proposed_dataset
            accepted = True
            current_utility = u_proposed
        else:
            accept_prob = self._acceptance_probability(delta_u)

            if random.random() < accept_prob:
                self.current_dataset = proposed_dataset
                accepted = True
                current_utility = u_proposed
            else:
                accepted = False
                current_utility = u_current

        # Update best seen
        if current_utility > self.best_utility:
            self.best_utility = current_utility
            self.best_dataset = self.current_dataset.copy()
            self.iterations_since_improvement = 0
        else:
            self.iterations_since_improvement += 1

        # Update temperature
        self.temperature = self._update_temperature()

        # Track history
        self.utility_history.append(current_utility)
        self.acceptance_history.append(accepted)

        # Increment iteration
        self.iteration += 1

        return ICMState(
            dataset=self.current_dataset.copy(),
            temperature=self.temperature,
            iteration=self.iteration,
            utility_score=current_utility,
        )

    def _should_stop_early(self) -> tuple[bool, str]:
        """Check if search should stop early.

        Returns:
            Tuple of (should_stop, reason).
        """
        # Stop if all examples are labeled
        if len(self.current_dataset) >= len(self.pool_data):
            return True, f"All {len(self.pool_data)} examples labeled"

        # Stop if temperature is very low and no recent improvement
        if (
            self.temperature < self.args.early_stop_threshold
            and self.iterations_since_improvement > self.args.early_stop_patience
        ):
            return (
                True,
                f"No improvement for {self.iterations_since_improvement} iterations "
                f"with T={self.temperature:.4f}",
            )

        # Stop if stuck with no improvement for long time
        if self.iterations_since_improvement >= self.args.early_stop_patience * 2:
            return (
                True,
                f"No improvement for {self.iterations_since_improvement} iterations",
            )

        return False, ""

    def run(self, verbose: bool = True) -> ICMResult:
        """Run ICM search with early stopping.

        Args:
            verbose: If True, show progress bar and log details.

        Returns:
            ICMResult with final dataset and history.
        """
        if verbose:
            logger.info(f"Starting ICM search with {self.args.sample_size} examples")
            logger.info(f"Max iterations: {self.args.max_iterations}")
            logger.info(f"Cooling rate (β): {self.args.cooling_rate}")
            logger.info(
                f"Temperature: {self.args.initial_temperature} -> {self.args.final_temperature}"
            )
            logger.info(f"Early stopping: patience={self.args.early_stop_patience}")

        pbar = tqdm(
            range(self.args.max_iterations),
            desc="ICM Search",
            disable=not verbose,
            ncols=100,
        )

        for i in pbar:
            state = self.step()

            # Update progress bar
            if verbose:
                pbar.set_postfix(
                    {
                        "size": len(state.dataset),
                        "utility": f"{state.utility_score:.3f}",
                        "best": f"{self.best_utility:.3f}",
                        "temp": f"{state.temperature:.2f}",
                        "acc_rate": f"{sum(self.acceptance_history) / len(self.acceptance_history):.0%}",
                    }
                )

            # Check early stopping
            should_stop, reason = self._should_stop_early()
            if should_stop:
                if verbose:
                    pbar.close()
                    logger.info(f"Early stopping at iteration {i + 1}: {reason}")
                break

        if verbose and not should_stop:
            pbar.close()

        return ICMResult(
            final_dataset=self.current_dataset,
            final_utility=self.utility_history[-1]
            if self.utility_history
            else -float("inf"),
            utility_history=self.utility_history,
            acceptance_history=self.acceptance_history,
            best_dataset=self.best_dataset,
            best_utility=self.best_utility,
        )
