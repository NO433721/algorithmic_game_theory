# Part A — From Nash Equilibrium to Maximin Strategies (two-player zero-sum)

**Setting and notation (explained inline).**  
We study a two-player *zero-sum* normal-form game. The row player’s (player 1’s) mixed strategy is written as \( \pi_1 \in \Pi_1 \), where \( \Pi_1 \) is the set of all probability distributions over her pure actions; analogously the column player’s (player 2’s) mixed strategy is \( \pi_2 \in \Pi_2 \). For any pair \( (\pi_1,\pi_2) \), let  
\[
u(\pi_1,\pi_2)
\]
denote the *row* player’s expected payoff; because the game is zero-sum, the column player’s payoff is \( -\,u(\pi_1,\pi_2) \). A **Nash equilibrium** is a pair \( (\pi_1^\*,\pi_2^\*) \) such that no unilateral deviation helps either player:
\[
u(\pi_1^\*,\pi_2^\*) \;\ge\; u(\pi_1,\pi_2^\*) \quad \forall\,\pi_1\in\Pi_1,
\qquad
-\,u(\pi_1^\*,\pi_2^\*) \;\ge\; -\,u(\pi_1^\*,\pi_2) \quad \forall\,\pi_2\in\Pi_2.
\]
Equivalently, \( \pi_1^\* \) is a best response to \( \pi_2^\* \) and \( \pi_2^\* \) is a best response to \( \pi_1^\* \).

The **maximin strategy** for the row player is any \( \hat\pi_1 \in \Pi_1 \) attaining
\[
\max_{\pi_1\in\Pi_1}\;\min_{\pi_2\in\Pi_2} u(\pi_1,\pi_2),
\]
i.e., it maximizes the worst-case (best-responding) opponent value. Symmetrically for the column player.

The **Minimax Theorem** (von Neumann) states
\[
\max_{\pi_1\in\Pi_1}\;\min_{\pi_2\in\Pi_2} u(\pi_1,\pi_2)
\;=\;
\min_{\pi_2\in\Pi_2}\;\max_{\pi_1\in\Pi_1} u(\pi_1,\pi_2)
\;=:\; v,
\]
and the common value \(v\) is the (unique) *game value*.

---

## Claim A (NE ⇒ Maximin).
If \( (\pi_1^\*,\pi_2^\*) \) is a Nash equilibrium, then each player’s equilibrium strategy is maximin; in particular,
\[
\pi_1^\* \in \arg\max_{\pi_1}\min_{\pi_2}u(\pi_1,\pi_2)
\quad\text{and}\quad
\pi_2^\* \in \arg\max_{\pi_2}\min_{\pi_1}(-u(\pi_1,\pi_2)).
\]
(We prove it for the row player; the column player is symmetric in zero-sum.)

### Proof.
1) **Best-response (saddle) inequalities from the NE conditions.**  
Because \( \pi_1^\* \) is a best response to \( \pi_2^\* \), for every \( \pi_1 \),
\[
u(\pi_1,\pi_2^\*) \;\le\; u(\pi_1^\*,\pi_2^\*).
\tag{BR\(_1\)}
\]
Because \( \pi_2^\* \) is a best response to \( \pi_1^\* \) *for the column*, it minimizes the row player’s payoff against \( \pi_1^\* \); hence for every \( \pi_2 \),
\[
u(\pi_1^\*,\pi_2^\*) \;\le\; u(\pi_1^\*,\pi_2).
\tag{BR\(_2\)}
\]
Together these say \( (\pi_1^\*,\pi_2^\*) \) is a *saddle point*:
\[
\forall\,\pi_1,\pi_2:\quad
u(\pi_1,\pi_2^\*) \;\le\; u(\pi_1^\*,\pi_2^\*) \;\le\; u(\pi_1^\*,\pi_2).
\]

2) **Identify the value realized at equilibrium.**  
From (BR\(_2\)) with \( \pi_2=\pi_2^\* \) we get
\[
\min_{\pi_2} u(\pi_1^\*,\pi_2) \;=\; u(\pi_1^\*,\pi_2^\*).
\tag{1}
\]
From (BR\(_1\)) we get, for every \( \pi_1 \),
\[
\min_{\pi_2} u(\pi_1,\pi_2) \;\le\; u(\pi_1,\pi_2^\*) \;\le\; u(\pi_1^\*,\pi_2^\*).
\tag{2}
\]

3) **Maximin optimality of \( \pi_1^\* \).**  
Taking the maximum over \( \pi_1 \) in the leftmost term of (2),
\[
\max_{\pi_1}\min_{\pi_2} u(\pi_1,\pi_2)
\;\le\;
u(\pi_1^\*,\pi_2^\*).
\tag{3}
\]
But by (1), \( \min_{\pi_2} u(\pi_1^\*,\pi_2) = u(\pi_1^\*,\pi_2^\*) \), hence
\[
\max_{\pi_1}\min_{\pi_2} u(\pi_1,\pi_2)
\;\ge\;
\min_{\pi_2} u(\pi_1^\*,\pi_2)
\;=\;
u(\pi_1^\*,\pi_2^\*).
\tag{4}
\]
Combining (3) and (4) gives
\[
u(\pi_1^\*,\pi_2^\*) \;=\; \max_{\pi_1}\min_{\pi_2} u(\pi_1,\pi_2).
\]
Thus \( \pi_1^\* \) attains the *maximin* value. By symmetry, \( \pi_2^\* \) is also maximin for the column player (equivalently, it attains \( \min_{\pi_2}\max_{\pi_1} u(\pi_1,\pi_2) \)). By the Minimax Theorem, these equal the common game value \(v\). ■

---

That completes the first implication. Want me to continue with **Part B — Maximin strategies ⇒ Nash equilibrium**, finishing the equivalence?
