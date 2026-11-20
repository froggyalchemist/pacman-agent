*Option 1*
- high carry limit (pacman tries to eat many pellets before returning home)
- pacman is very scared of ghosts, he will run away as soon as he sees one

*Option 2*
- low carry limit -> pacman returns home very often to drop pellets
- not too scared of ghosts -> runs away only when they're very close


Problems with OffensiveQAgent:
 - looks for nearest food and gets stuck when fleeing
 - when it eats the power pellet it doesn't try to eat ghosts --> feature that tells pacman whether ghosts are scared
 - gets stuck in dead ends A LOT, he isn't aware
- sometimes it gets stuck in the middle looking back and forth --> look at previous two or three actions instead of looking only at the last one
