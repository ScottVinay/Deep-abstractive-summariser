# Deep-abstractive-summariser

This is a project I built to test my hand at automatic sequence-to-sequence summarisation. This was trained on WikiHow articles, with the aim of predicting a section's headline from the main text. Some decent results can be seen below.

## Novel contributions

Here I will note some things I have included that may not be found in other implementations of deep translators, so that they may help anyone else trying a similar problem.

• I noticed that many of the sentences produced were too short. This was because the end token was more common than most other words, and appeared in more diverse contexts, so was predicted more often than it should be. To counter this, I included a suppression hyperparameter, which suppressed the end token by some factor, _F_.

• In order to choose the best value for _F_, a metric was needed to determine how close a test-set prediction was in meaning to the actual headline. Simply using accuracy on the predicted word set was not appropriate, so I hand-coded in a BLEU score to act as a metric for hyper-parameter optimisation.

• I coded in a beam-translator, but I noticed that by the definitions of the Bayesian probabilities, it was likely to favour short sentences. I included an optional parameter to make it so that the probability of a given beam was conditional on the probability that a sentence of that length was found.

====================================================================================================

## Some sample outcomes:

__Text:__ The minimum requirements to become a licensed real estate agent in the state of montana are that you must be at least 18 years of age and have a high school diploma, or an equivalent, such as a ged. If you do not have a high school diploma or a an equivalent, you will need to obtain a ged before completing your pre license educationcommunity colleges offer classes that will help prepare you to pass the ged

__Actual headline:__ Obtain your ged

__Generated headlines:__

• Get a license

• Get a degree

• Get a college degree

--------------------------------------------------------------------------------------------------------------

__Text:__ Stock up well before thanksgiving so that you don't have to do all of your shopping at once. Things like pumpkin pie filling or canned cranberry sauce are more readily available before november and can be purchased at most grocery stores. Things like plastic storage bags, aluminum foil, and sealable storage containers are also great things to pick up in advance.

__Actual headline:__ Purchase non-perishable food early.

__Generated headlines:__
    Gather your supplies
    Make a list of items you need
    Make a list of items you want to use

Text: A large part of being mature and responsible enough to live on your own is understanding how much money you will have going out, what youll have coming in, and how much is left over at the end. A budget can be extremely useful, because then you know exactly where your money is going, and whether you can afford certain things or notfor instance, if you know your budget only allots 40 for food per week, you will immediately know that you shouldnt spend 10 of that on a single fast food meal.

Actual headline: Create a budget

Generated headlines:

    Make a budget
    Make a budget for your budget
    Make a list of your budget

Text: Heat and cold can help to relieve pain and muscle tension in your neck and headapply a moist hot towel or warm compress to the back of your neck or on your forehead. You can also take a long, hot shower, being sure to run water down your head and on the back of your neck. Wrap an ice pack in a towel and place it on the back of your neck or on your forehead.

Actual headline: Apply a hot or cold compress to your head

Generated headlines:

    Take a shower
    Take a warm bath
    Take a warm shower

