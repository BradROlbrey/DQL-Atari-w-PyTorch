For the final project of my artificial intelligence class, I tried to recreate Google Deepmind's [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/pdf/1312.5602v1.pdf). Long story short, I don't really succeed, as they were able to do the infamous tunneling technique in Atari Breakout. I too had hoped to achieve this, however my agent never achieves such grand aspirations and actually performs rather inconsistently. However, it does show some learning potential, as the score counter does go up over time across a few different games and it does clearly get better at playing.

In Breakout, it sometimes plays well and gets a high score, like so:

![Breakout_good](https://user-images.githubusercontent.com/17125101/214050157-c554e05d-cd22-442e-82af-b47f7b17260c.gif)

But most of the time it flubs, like this:

![Breakout_bad](https://user-images.githubusercontent.com/17125101/214050132-dd200aa1-ec19-4af1-ba35-da3d78135e46.gif)

I think I like watching it play Space Invaders the most:

![SpaceInvaders](https://user-images.githubusercontent.com/17125101/214050058-2d17e355-48f0-4334-8737-12821dfc05f6.gif)

In Pong it appears to perform decently well, though it's mostly just gaming the system with how it hits the ball back the same way at the start of each round and typically loses the round when that doesn't work.

![Pong](https://user-images.githubusercontent.com/17125101/214050103-ddc7a63b-802d-4448-8f14-02ee738c3bb8.gif)


These are plots for each game showing the average-score-of-the-past-few-games vs frames-played. The score goes up in the beginning as the agent learns how to play, but then the score plateaus. In the case of Pong, this is simply because 20 points is the maximum achievable score. But for Breakout, it's because the agent never achieves true mastery, sometimes scoring very high but oftentimes playing abysmally throughout training. As for Space Invaders, I'd say it plays that game pretty well.

![All](https://user-images.githubusercontent.com/17125101/214060910-41caeab9-6488-4983-a493-95ff4085fa0a.jpg)

It tends to get stuck in patterns, which are especially visible in the way it plays Breakout and Pong. I've likely done some things wrong and there are certainly things that could be improved. However, I'm pretty proud of what I created as is, and hopefully one day I will come back to work on it and make it even better.

