MSA tells how evolution works -- reference thats update
Pair grid is output of model, how confident different pairs go in contact

* row attention
  * update your MSA sequences + bias from pair grid

* column attention
  * normal attention along different species (no bias here since pair grid isn't inter-species etc, just inter-residues)

* (then feedforward step)

* Outer product
  * take the attended msa representations: position 1 vector outer product w/ position 2 (each entry in vector is for certain species)
  * average them so high correlation signal gets exposed
  * fit into small learned function, linear layer, that squashes into what pair grid needs
  * add this value to the pair grid entry

TRIANLE ATTEENTION AND UPDATE:

* So far each pair cell has been computed on its own, so we now add in triangle updates
* But remember this is 3d geometry
* If three points sit somewhere in 3D, and two of the three pairs are close together, the third pair cannot be arbitrarily far apart.
* So when the pair grid says "1 and 2 close, 2 and 4 close, 1 and 4 far," that's not just a weird disagreement between three numbers, it's a claim about geometry that cannot correspond to any real arrangement of three points anywhere. It's describing something that literally cannot be built.
* Triangle update says that we should update it so that we can ultimately output real coords

Triangle update: every third point k gets treated the same way, using one fixed rule. It's like averaging opinions from everyone in the room, using the same method every time, regardless of who's talking or what the topic is.

Triangle attention: instead of treating every k the same, it asks "for this specific pair I'm updating, which third points actually matter most right now" and weighs them accordingly, differently each time depending on the situation.


Triangle update, updating cell(i,j) using every third point k:

z_ij_new = z_ij + sum over k of [ a_ik × b_jk ]

where:
  a_ik = gate_a(z_ik) × linear_a(z_ik)
  b_jk = gate_b(z_jk) × linear_b(z_jk)

Each term in that sum only depends on k, i, and j through the gates and linear layers, the same fixed functions every time. Nothing in this formula compares k to the specific pair (i,j) beyond just multiplying the two edges together. Every k contributes to the sum using the identical rule.

Triangle attention, updating the same cell(i,j):

score(i,j,k) = (q_ij · k_ik) / sqrt(d) + bias(z_jk)
weight(i,j,k) = softmax over k of score(i,j,k)
z_ij_new = sum over k of [ weight(i,j,k) × v_ik ]

The key difference is sitting right in score(i,j,k): it's a dot product between something built from the specific pair (i,j) being updated, q_ij, and something built from each candidate third point, k_ik. That dot product is different for every k, and it's different depending on which (i,j) you're updating. The softmax then turns those scores into weights that sum to 1, some k's get a big slice, some get almost none, and which ones win depends on the actual content of z_ij, not a fixed rule.


Why not just attention? 

Attention normalizes, update doesn't — softmax weights sum to 1, so attention can only average, never amplify. Update can let many agreeing third points stack into a stronger signal than any single one.
Attention needs to be trained to be useful, update doesn't — early in training, query/key projections are near-random, so attention weights are close to uniform. Update's multiplicative rule works from the start.
Attention is more expensive — extra query/key projections and a softmax, run for every triple, every block. Update is cheaper and gives cheap structural consistency; attention then refines on top of it.
Empirically, both together beat either alone — not fully derivable from first principles, ablations just showed it works better.


Then another feedforward 48x


-----

Structure module

Input to the Structure Module: the pair grid (48 blocks of Evoformer refinement, holding pairwise beliefs) and the single representation, each residue's own feature vector, pulled from the MSA track.

What IPA computes, per residue pair (i,j):

A content score, ordinary Q·K dot product between residues' feature vectors, same mechanism as row attention.
A pair-grid bias, pulled from the pair grid, same trick row attention used.
A geometric score, built from real 3D points: each residue's feature vector generates a local point, that point gets placed in global space using the residue's own current frame (R_i, t_i, its rotation and position, already tracked from previous passes), and the score is based on distance between those global points.

All three get summed into one score, softmaxed into attention weights, exactly like every other attention step in this whole model.

1. Side chains get placed. The backbone frames are done, now each residue predicts a handful of angles, chi angles, and side chains snap onto the backbone using those angles plus fixed bond geometry. Much cheaper than the backbone problem, this is basically a lookup-and-rotate, not another attention mechanism.

2. Confidence heads run. pLDDT per residue, PAE per pair, both looking at this now-finished structure and estimating how much to trust it, exactly the metrics you covered in real depth earlier.

3. The result feeds back up, not just within the Structure Module, but to the whole model. Remember the very first architecture map, the outer recycling loop, sequence → Evoformer → Structure Module → back into Evoformer. This is that. The coordinates just produced, plus how confident the model was in them, get fed back in as extra input to the next full pass through Evoformer, not just another IPA loop, an entire additional trip through all 48 blocks, MSA and pair grid both, now informed by an actual real structure the model built on the previous attempt.
