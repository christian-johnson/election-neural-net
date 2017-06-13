# Were Vote Counts in the 2016 US Presidential Election Hacked?
At this point, you would be forgiven for not wanting to read yet another article about the the Trump-Russia affair.
But in recent days a few things have come to light, revealing that the scope of Putin's campaign of interference was even broader than previously thought. The new revelations are especially troubling because they imply that the Russians didn't just try to influence votes through a sophisticated propaganda machine, but that the very results of the election are in question.
So what are the new revelations?

On June 6 2017, NSA contractor [Reality Winner](https://www.theatlantic.com/news/archive/2017/06/who-is-reality-winner/529266/) was arrested on suspicion of leaking a top-secret document to The Intercept.
The document [alleges](https://theintercept.com/2017/06/05/top-secret-nsa-report-details-russian-hacking-effort-days-before-2016-election/) that Russia attempted to spear-phish up to 100 local election officials in up to seven states, with the intention of gaining access to their computer systems.
A week later, [Bloomberg](https://www.bloomberg.com/news/articles/2017-06-13/russian-breach-of-39-states-threatens-future-u-s-elections) reported that Russian hackers had attacked electronic voting infrastructure in up to 39 states. It goes without saying that if the Russians were trying to alter vote counts to get Donald Trump elected as president, the American people deserve to know.
Hopefully law enforcement and intelligence agencies will perform a thorough investigation and make their findings public- but that could take a very long time.

I wondered whether it is possible to detect voting interference without having direct access to the voting machines themselves. To find out, we need to start by putting ourselves in the mindset of Vladimir Putin, and assume that he has the capability and determination to pull this off.
## So you want to hack an election
For obvious reasons, you wouldn't want to get caught changing vote tallies. That means Putin can't use any of his usual shenanigans, like bussing people around to multiple voting sites. This effort has got to be more subtle- and that means tampering with electronic voting machines. 
You also would want to concentrate your efforts on only a handful of swing states- it doesn't make much sense to try to swing California for Trump, for example. Your risk of getting caught also increases as you expand the number of targets. So if I were Putin, I would focus on states like Florida, North Carolina, Ohio, and Pennsylvania, which were the biggest swing states this election. In hindsight, we know that Michigan and Wisconsin also had razor-thin margins and ultimately voted in favor of Trump, but in the weeks leading up to the election those states were widely considered safe ground for the Democrats.

Within each state, the same principle of limiting your targets still applies- only this time the only thing that matters is the popular vote within the state. Therefore it makes sense to target the big population centers within each swing state (a few tens of thousands of votes changed one way or another will go unnoticed in Charlotte, for example, but would be highly conspicuous in a small county).

Ok, so we can make an educated guess that the places most likely to be a target for vote tampering are counties that contain large-ish cities, in swing states. The big question still remains- how would we know if the votes counts in those counties are valid or not? What we need is an accurate, county-by-county prediction of the Trump vote fraction, which we can then compare to the actual votes. To do this, we'll turn to machine learning.

## Crunching numbers with neural nets



{% highlight python %}
from sklearn.neural_network import MLPRegressor as mlp
{% endhighlight %}
