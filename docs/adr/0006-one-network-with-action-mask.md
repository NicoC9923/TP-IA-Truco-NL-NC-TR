# One network decides every action, with a legal-action mask

A single network has one output per possible action: play card 1, 2 or 3; envido, real envido and falta envido; quiero and no quiero; truco, retruco and vale cuatro; and folding (irse al mazo). At each decision, illegal actions are masked out and the choice is made among the rest. This was preferred over three separate networks (envido, truco and cards) because it shares what is learned across decisions (hand strength matters both for calling truco and for choosing a card) and doesn't triple the number of weights to evolve.
