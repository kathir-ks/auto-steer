"""
Auto-steer v2: preparation and utility script.

One-time setup: load Gemma 2 2B, generate contrastive concept prompts,
run forward passes capturing residual-stream activations at every layer,
and cache everything to disk as numpy arrays.

Uses PyTorch for model loading/inference. Analysis in steer.py uses
numpy/sklearn (and optionally JAX for TPU-accelerated computations).

Usage:
    python3 prepare_steer.py                # Full setup (prompts + extract)
    python3 prepare_steer.py --prompts-only # Only regenerate concept prompts
    python3 prepare_steer.py --extract-only # Only re-extract activations
"""

import os
import json
import argparse
import time
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "google/gemma-2-2b"
MAX_SEQ_LEN = 128          # tokens per prompt
PROMPTS_PER_DIRECTION = 60  # prompts per positive/negative side per concept
EXTRACTION_POSITIONS = ["last", "mean"]  # extract both last-token and mean-pool

CACHE_DIR = Path.home() / ".cache" / "autosteer-v2"
PROMPTS_PATH = CACHE_DIR / "concept_prompts.json"
ACTIVATIONS_DIR = CACHE_DIR / "activations"
META_PATH = CACHE_DIR / "extraction_meta.json"

# ---------------------------------------------------------------------------
# Concept categories — contrastive prompt pairs
# Each concept has "positive" and "negative" directions with example prompts.
# The agent does NOT modify these; they are fixed inputs to the analysis.
# ---------------------------------------------------------------------------

CONCEPTS = {
    "sentiment": {
        "description": "Positive vs negative emotional tone",
        "positive": [
            "I am so happy today, everything feels wonderful and bright.",
            "This is the best day of my life, I feel incredibly grateful and blessed.",
            "The sunset was breathtaking and filled me with pure joy.",
            "What a fantastic meal, every bite was absolutely delicious.",
            "I love spending time with my family, they bring me so much happiness.",
            "The concert was incredible, the music moved me to tears of joy.",
            "I got the promotion I have been working towards for years!",
            "My garden is blooming beautifully this spring, it fills my heart.",
            "The children were laughing and playing, it was a heartwarming scene.",
            "I finally finished the marathon and the feeling of accomplishment is amazing.",
            "Every morning I wake up feeling thankful for this beautiful life.",
            "The surprise birthday party was perfect, everyone I love was there.",
            "Our team won the championship and the celebration was euphoric.",
            "The puppy greeted me at the door with its tail wagging excitedly.",
            "I received the kindest letter from an old friend today.",
            "The view from the mountaintop was absolutely spectacular and uplifting.",
            "My students passed their exams with flying colors, I am so proud.",
            "The wedding ceremony was beautiful, filled with love and laughter.",
            "I discovered a charming little cafe that serves the most wonderful pastries.",
            "After months of hard work, the project was a tremendous success.",
            "The baby smiled at me for the first time and my heart melted.",
            "Spring has arrived and the whole world feels fresh and alive.",
            "I reconnected with an old friend and it felt like no time had passed.",
            "The book I just finished was one of the most inspiring I have ever read.",
            "My recovery is going well and I feel stronger every single day.",
            "The community came together to help and it restored my faith in people.",
            "I just adopted the sweetest rescue cat and she already loves her new home.",
            "The holiday decorations made the whole neighborhood feel magical.",
            "Watching the sunrise over the ocean was a profoundly peaceful experience.",
            "I am so grateful for all the opportunities that have come my way.",
            "The laughter of children echoing through the house is the sweetest sound.",
            "We celebrated our anniversary with a lovely dinner under the stars.",
            "The kindness of strangers during my travels restored my optimism.",
            "I feel so alive after that invigorating morning run through the park.",
            "The orchestra played with such passion that it gave me chills of delight.",
            "My best friend surprised me with tickets to my favorite show.",
            "The smell of fresh bread baking reminds me of my happiest childhood days.",
            "I am thrilled to announce that our charity exceeded its fundraising goal.",
            "The warm hug from my mother made everything feel right in the world.",
            "Seeing the Northern Lights was one of the most magical experiences of my life.",
            "The entire team celebrated together and the energy was absolutely electric.",
            "I feel incredibly lucky to have such supportive and caring colleagues.",
            "The first snowfall of winter turned the town into a fairytale landscape.",
            "Volunteering at the shelter filled me with a deep sense of purpose and joy.",
            "My painting was selected for the exhibition and I could not be more proud.",
            "The sunset painted the sky in shades of pink and gold, it was heavenly.",
            "Finding that lost heirloom in the attic brought tears of happiness to my eyes.",
            "The roaring applause after my presentation was the most rewarding feeling.",
            "I woke up to breakfast in bed prepared by my thoughtful partner.",
            "The gentle sound of waves lapping the shore is pure tranquility.",
            "Our garden party was filled with music, good food, and genuine laughter.",
            "I passed my driving test on the first attempt and I am over the moon.",
            "The handwritten thank-you note from a student touched my heart deeply.",
            "Watching my child take their first steps was an unforgettable moment of pure joy.",
            "The crisp autumn air and colorful leaves make every walk a delight.",
            "I received a glowing performance review and a well-deserved raise today.",
            "The cozy cabin retreat with friends was exactly what my soul needed.",
            "Hearing my favorite song unexpectedly on the radio made my whole day brighter.",
            "The standing ovation after the school play made every rehearsal worthwhile.",
            "I feel blessed to live in such a beautiful and peaceful corner of the world.",
        ],
        "negative": [
            "I am so sad today, everything feels terrible and hopeless.",
            "This is the worst day of my life, I feel incredibly miserable and alone.",
            "The weather was dreadful and filled me with a deep sense of despair.",
            "What an awful meal, every bite was disgusting and unpleasant.",
            "I hate being stuck in traffic, it makes me so frustrated and angry.",
            "The concert was disappointing, the sound was awful and the crowd was rude.",
            "I lost the job I had been counting on and feel completely devastated.",
            "My garden is dying because of the drought and it breaks my heart.",
            "The children were fighting and screaming, it was an exhausting nightmare.",
            "I failed the exam despite studying for weeks and feel utterly worthless.",
            "Every morning I wake up dreading what the day will bring.",
            "Nobody remembered my birthday and I spent the evening alone crying.",
            "Our team lost badly and the defeat felt crushing and humiliating.",
            "The dog destroyed my favorite shoes and I lost my temper completely.",
            "I received a rejection letter from the university I dreamed of attending.",
            "The view from my apartment is just a grey concrete wall.",
            "My students failed the test and I feel like I let them down.",
            "The funeral was devastating, saying goodbye was the hardest thing ever.",
            "I went to the restaurant and the service was rude and the food was cold.",
            "After months of effort, the project failed spectacularly.",
            "The baby would not stop crying all night and I am completely exhausted.",
            "Winter drags on endlessly and everything feels cold and lifeless.",
            "I lost touch with my closest friend and the loneliness is unbearable.",
            "The book was depressing and left me feeling empty inside.",
            "My health is declining and every day feels harder than the last.",
            "Nobody came to help when I needed it most, I felt abandoned.",
            "The shelter had to turn away animals and it was heartbreaking to see.",
            "The neighborhood feels unsafe and unwelcoming after dark.",
            "Watching the news filled me with anxiety and a sense of helplessness.",
            "I feel trapped by circumstances and see no way forward.",
            "The constant noise from construction next door is driving me to despair.",
            "I missed the last train home and spent a miserable night at the station.",
            "The argument with my partner left me feeling drained and heartbroken.",
            "My savings were wiped out by unexpected medical bills and I feel crushed.",
            "The grey monotony of this town makes every day feel the same.",
            "I was passed over for the promotion again despite all my hard work.",
            "The smell of the polluted river makes the whole area depressing to live in.",
            "Our fundraiser failed miserably and I feel like I have let everyone down.",
            "The cold silence from my family during the holidays was deeply painful.",
            "Seeing my childhood home demolished left me with an aching sense of loss.",
            "The hostile atmosphere at work is making me dread every single morning.",
            "I feel completely invisible to my coworkers and it hurts more than I admit.",
            "The relentless rain has turned the garden into a muddy wasteland.",
            "Being turned away from the shelter on a freezing night was soul-crushing.",
            "My artwork was rejected from the exhibition and my confidence is shattered.",
            "The dark, gloomy skies seem to mirror the sadness I carry inside.",
            "Losing that precious family heirloom to the flood was absolutely devastating.",
            "The harsh criticism after my presentation left me feeling humiliated.",
            "I woke up to another empty house and the silence was overwhelming.",
            "The relentless crashing of waves during the storm kept me awake with dread.",
            "Our garden was destroyed by vandals and it broke the spirit of the whole street.",
            "I failed my driving test for the third time and I am losing all hope.",
            "The dismissive response from the teacher made me feel small and stupid.",
            "Watching my elderly parent struggle with simple tasks fills me with sorrow.",
            "The bare trees and bitter cold of winter make everything feel dead and empty.",
            "I received a terrible performance review and now I fear for my future.",
            "The cramped and noisy hostel made the trip an exhausting ordeal.",
            "Hearing that song reminds me of everything I have lost and it aches.",
            "The awkward silence after the failed joke made me want to disappear.",
            "I feel cursed to live in such a bleak and forgotten part of the world.",
        ],
    },
    "formality": {
        "description": "Formal professional vs casual informal register",
        "positive": [
            "I am writing to formally request an extension on the quarterly report deadline.",
            "The committee hereby approves the proposed amendments to the charter.",
            "We would like to express our sincere gratitude for your generous contribution.",
            "Pursuant to the agreement, all deliverables shall be submitted by the deadline.",
            "The board of directors convened to discuss the fiscal year projections.",
            "It is with great pleasure that we announce the appointment of the new director.",
            "Please find enclosed the documentation pertaining to your recent inquiry.",
            "The organization maintains strict adherence to all regulatory requirements.",
            "We respectfully request that all attendees observe the established protocol.",
            "The comprehensive analysis indicates a statistically significant correlation.",
            "In accordance with company policy, all employees must complete the training.",
            "The distinguished panel of experts will present their findings at the conference.",
            "Herein we outline the methodology employed in the aforementioned study.",
            "Your prompt attention to this matter would be greatly appreciated.",
            "The institution has implemented enhanced measures to ensure compliance.",
            "We acknowledge receipt of your correspondence dated the fifteenth of March.",
            "The undersigned parties agree to the terms and conditions specified herein.",
            "It is imperative that all personnel adhere to the revised safety guidelines.",
            "The empirical evidence substantiates the hypothesis presented in the report.",
            "We are pleased to confirm your reservation for the annual symposium.",
            "The executive summary provides an overview of the key findings and recommendations.",
            "All candidates must demonstrate proficiency in the required competencies.",
            "The proceedings of the tribunal shall be conducted in accordance with the rules.",
            "We extend our warmest congratulations on this remarkable achievement.",
            "The strategic initiative aims to optimize operational efficiency across divisions.",
            "Formal approval must be obtained prior to the commencement of the project.",
            "The quarterly earnings report reflects sustained growth in all market segments.",
            "We are obligated to inform you of the changes to the terms of service.",
            "The delegation expressed their commitment to fostering bilateral cooperation.",
            "This memorandum serves to clarify the procedures outlined in the previous directive.",
            "The aforementioned clauses shall remain in effect for the duration of the contract.",
            "We respectfully submit the enclosed proposal for your consideration and review.",
            "The regulatory framework mandates full disclosure of all material information.",
            "It is incumbent upon all stakeholders to exercise due diligence in this matter.",
            "The auditor's report confirms that all financial statements are presented fairly.",
            "We hereby certify that the information contained herein is accurate and complete.",
            "The resolution was adopted unanimously by the assembled members of the council.",
            "Kindly direct all further inquiries to the designated liaison officer.",
            "The fiduciary responsibility of the trustees extends to all beneficiaries of the fund.",
            "In light of recent developments, the committee recommends a thorough reassessment.",
            "The terms of reference for the review panel have been duly established.",
            "We wish to draw your attention to the provisions set forth in Section twelve.",
            "The performance metrics demonstrate consistent improvement across all key indicators.",
            "All correspondence should be addressed to the office of the registrar.",
            "The curriculum has been revised in accordance with the latest accreditation standards.",
            "We take this opportunity to reaffirm our commitment to ethical governance.",
            "The statistical analysis was conducted using internationally recognized methodologies.",
            "Participants are advised to familiarize themselves with the code of conduct.",
            "The consortium has agreed to allocate additional resources to the initiative.",
            "We look forward to a mutually beneficial and enduring professional relationship.",
            "The symposium proceedings will be published in the official journal of the society.",
            "All amendments to the bylaws require a two-thirds majority for ratification.",
            "The compliance officer shall oversee the implementation of the new regulations.",
            "We respectfully decline the invitation due to prior commitments.",
            "The longitudinal study provides robust evidence supporting the proposed intervention.",
            "It is our pleasure to present the findings of the comprehensive review.",
            "The secretary shall record the minutes of each meeting for official archive.",
            "We commend the exemplary dedication demonstrated by all members of the task force.",
            "The proposal has been evaluated against the established selection criteria.",
            "Formal notice is hereby given of the forthcoming changes to the operating procedures.",
        ],
        "negative": [
            "Hey dude, can you push back the deadline? I'm swamped lol.",
            "Yeah so basically we said cool let's just change the rules a bit.",
            "Thanks a ton for the cash, you're seriously the best!",
            "Gotta get that stuff turned in on time or we're toast haha.",
            "The bosses got together and talked about next year's money stuff.",
            "Guess what, we got a new boss! Pretty exciting right?",
            "Here's the stuff you asked about, take a look when you get a sec.",
            "We're super strict about following the rules and all that jazz.",
            "Hey everyone, please just like, follow the plan okay?",
            "So basically the data shows these two things are totally connected.",
            "Company says everyone's gotta do the training thing, no exceptions.",
            "A bunch of smart people are gonna talk at the conference next week.",
            "So here's how we did the thing I was telling you about.",
            "Can you deal with this ASAP? Would really help me out.",
            "We've beefed up security or whatever to make sure we're good.",
            "Got your email from last week, my bad for the late reply!",
            "We all agreed on the deal, shook hands and everything.",
            "Yo, everyone needs to follow the new safety stuff, no joke.",
            "The numbers pretty much prove what we thought all along.",
            "You're all set for the event next month, see ya there!",
            "Here's the quick version: we found some cool stuff and have ideas.",
            "You gotta show you know your stuff to get the gig.",
            "The court thing is gonna go by the book, don't worry.",
            "Congrats!! That's so awesome, you totally deserve it!",
            "We're trying to make things run smoother across the whole company.",
            "Just get the OK before you start working on it.",
            "Money's looking good this quarter, we're up across the board.",
            "Heads up, we changed some stuff in the terms and conditions.",
            "The folks from the other country said they wanna work together more.",
            "Just wanted to clear up the confusion from that last email.",
            "Those rules from the contract still apply for now btw.",
            "We threw together a proposal, lemme know what you think.",
            "Rules say you gotta spill the beans on all the important stuff.",
            "Everyone involved needs to do their homework on this one.",
            "The money people checked everything and say the books look fine.",
            "We promise everything in here is legit and on the up and up.",
            "Everyone voted yes, not a single nope in the bunch.",
            "Hit up the contact person if you have more questions.",
            "The people in charge of the money have to look out for everyone.",
            "Things have changed so the team thinks we should take another look.",
            "We've set up what the review group is supposed to be doing.",
            "Just FYI check out what it says in part twelve of the doc.",
            "The numbers show we're getting better at basically everything.",
            "Send any letters or whatever to the main office.",
            "They updated the classes to match the new standards and stuff.",
            "Just wanna say again that we're all about doing the right thing.",
            "They crunched the numbers using the methods that everyone agrees on.",
            "Make sure you read the rules before showing up, yeah?",
            "The group agreed to throw more money and people at the project.",
            "Looking forward to a solid working relationship, should be great.",
            "They're gonna publish what happened at the conference in the journal.",
            "If you wanna change the rules you need like most people to agree.",
            "The rules person is gonna make sure everyone follows the new stuff.",
            "We're gonna have to pass on that invite, got other plans sorry.",
            "The long study basically proves the new approach works pretty well.",
            "Happy to share what we found in the big review.",
            "Someone's gotta write down what happens in the meetings for the records.",
            "Shoutout to everyone on the team, you guys killed it!",
            "We checked the proposal against all the boxes and it looks solid.",
            "Just a heads up, some changes are coming to how we do things.",
        ],
    },
    "certainty": {
        "description": "Confident assertions vs uncertain hedging",
        "positive": [
            "This is absolutely the correct approach, there is no doubt about it.",
            "I am completely certain that the deadline will be met on schedule.",
            "The evidence clearly demonstrates that this treatment is highly effective.",
            "Without question, this is the best solution available to us right now.",
            "I guarantee that these results are accurate and reproducible.",
            "There is no ambiguity here: the data unequivocally supports our conclusion.",
            "I am one hundred percent confident in the accuracy of this analysis.",
            "The facts speak for themselves, this policy has definitively succeeded.",
            "We know exactly what caused the failure and have already fixed it.",
            "This will undoubtedly be the most significant breakthrough of the decade.",
            "I can assure you that every detail has been thoroughly verified.",
            "The answer is clear and indisputable: we must proceed immediately.",
            "There is zero chance of failure if we follow this plan precisely.",
            "I am firmly convinced that this is the right direction for the company.",
            "The conclusion is inescapable: the old method is clearly inferior.",
            "We have definitive proof that the system works as designed.",
            "I stake my reputation on the correctness of these findings.",
            "It is an established fact that this approach yields superior results.",
            "The outcome is predetermined, there is nothing that can change it now.",
            "I know with absolute certainty what went wrong and how to fix it.",
            "This strategy will succeed, I have no reservations whatsoever.",
            "The research conclusively proves the hypothesis beyond any reasonable doubt.",
            "We are fully confident in the security of our infrastructure.",
            "I can state categorically that no errors were made in the process.",
            "The trajectory is unmistakable: we are headed for record performance.",
            "This is proven technology that has been validated thousands of times.",
            "I personally verified every single data point in this report.",
            "There is absolutely no risk associated with this particular decision.",
            "The team is entirely aligned and there is unanimous agreement on the plan.",
            "Make no mistake, this is the defining moment for our organization.",
            "I am positive beyond all doubt that the experiment will be replicated.",
            "The law of physics guarantees that this engine will outperform all rivals.",
            "Our testing has been exhaustive and the product is flawless.",
            "Every credible expert agrees: this is the gold standard in the field.",
            "I will stake my career on the claim that these projections are accurate.",
            "There is irrefutable proof sitting right here on the laboratory bench.",
            "The numbers leave no room for interpretation, we have achieved our target.",
            "I have never been more sure of anything in my entire professional life.",
            "This discovery will inevitably reshape the entire industry within five years.",
            "The audit trail confirms, without exception, that every protocol was followed.",
            "We possess overwhelming evidence that the competitor's claims are false.",
            "The correlation is undeniable and the causal mechanism is well understood.",
            "I swear on everything I hold dear that these measurements are correct.",
            "Nothing could convince me otherwise, the data is crystal clear.",
            "This is not speculation, it is established scientific consensus.",
            "Our model predicts the outcome with perfect accuracy every single time.",
            "I have absolute faith in the integrity of this research team.",
            "The verdict is final and there are no grounds for appeal whatsoever.",
            "We have categorically ruled out every alternative explanation.",
            "This framework is bulletproof and has withstood every challenge thrown at it.",
            "I am dead certain that the root cause has been identified and resolved.",
            "The signal in the data is unmistakable, there is no noise to speak of.",
            "Every single test we have run confirms the original hypothesis.",
            "There is literally zero ambiguity in what these results are telling us.",
            "I would bet my life savings that this prediction turns out to be correct.",
            "The proof is mathematically rigorous and has been independently verified.",
            "I have personally inspected every component and they are all perfect.",
            "The trend is crystal clear and anyone who denies it is ignoring reality.",
            "We are one hundred percent certain that the vulnerability has been patched.",
            "History will prove us right, I have absolutely no doubt about that.",
        ],
        "negative": [
            "I think this might be the right approach, but I am not entirely sure.",
            "The deadline could potentially be met, though there are some uncertainties.",
            "The evidence seems to suggest this treatment might be somewhat effective.",
            "This could possibly be a good solution, but we should consider alternatives.",
            "I believe these results are probably accurate, but more testing may be needed.",
            "It appears that the data may support our conclusion, to some extent.",
            "I am fairly confident, though I would not rule out the possibility of errors.",
            "The policy seems to have had some degree of success, arguably.",
            "We think we might have identified the cause, but we are still investigating.",
            "This could potentially turn out to be an important development, perhaps.",
            "I hope everything has been checked, but there may be gaps we missed.",
            "It seems like we should probably proceed, but I have some reservations.",
            "There is probably a low chance of failure, though nothing is guaranteed.",
            "I am leaning towards this direction, but I could be persuaded otherwise.",
            "The old method might be less effective, though there are trade-offs.",
            "We have some evidence suggesting the system may work as intended.",
            "I am reasonably confident, but I would not bet everything on it.",
            "It is generally believed that this approach tends to produce better results.",
            "The outcome is somewhat uncertain and could go either way at this point.",
            "I have a hunch about what went wrong, but I need to look into it more.",
            "This strategy might succeed, though there are risks I am worried about.",
            "The research provides preliminary support for the hypothesis.",
            "We are somewhat confident in our security, but vulnerabilities may exist.",
            "I believe the process was mostly sound, though I cannot be completely sure.",
            "The trend looks promising, but it is too early to draw firm conclusions.",
            "This is relatively well-tested technology, although edge cases remain.",
            "I reviewed most of the data points, but some still need verification.",
            "There is some risk involved, and we should have contingency plans.",
            "Most of the team seems to agree, though there are a few dissenting views.",
            "This may be an important moment, but only time will truly tell.",
            "I suspect the experiment could be replicated, but I would not guarantee it.",
            "The engine might outperform rivals, although real-world conditions may differ.",
            "Our testing has been fairly thorough, though some scenarios remain untested.",
            "Many experts lean this way, but the consensus is not yet fully settled.",
            "I am fairly optimistic about these projections, though margins of error exist.",
            "There appears to be suggestive evidence, but it is far from conclusive.",
            "The numbers look encouraging, though interpretation is somewhat subjective.",
            "I feel reasonably sure, but I have been wrong about things like this before.",
            "This discovery could possibly reshape parts of the industry over time.",
            "The audit suggests protocols were mostly followed, with a few minor gaps.",
            "We have some indications that the competitor's claims may be overstated.",
            "There seems to be a correlation, but the causal link is still debated.",
            "I am fairly confident these measurements are in the right ballpark.",
            "It would take strong evidence to change my mind, but I remain open.",
            "This is an emerging area where the science is still evolving.",
            "Our model does well in most cases, though outliers occasionally appear.",
            "I have reasonable confidence in the team, though nobody is infallible.",
            "The ruling seems sound, but there may be grounds for further review.",
            "We have ruled out several alternatives, though a few possibilities linger.",
            "The framework has held up well so far, but new challenges may emerge.",
            "I am fairly certain the root cause has been found, but I want to double-check.",
            "There appears to be a signal in the data, though noise complicates things.",
            "Most of our tests support the hypothesis, with a few inconclusive results.",
            "There is relatively little ambiguity, though some edge cases are unclear.",
            "I would be willing to wager that this prediction is probably correct.",
            "The proof seems solid, though peer review may surface minor issues.",
            "I have inspected most components and they appear to be in good condition.",
            "The trend seems fairly clear, though alternative readings are possible.",
            "We believe the vulnerability is patched, but ongoing monitoring is prudent.",
            "I suspect history will look favorably on this decision, though who can say.",
        ],
    },
    "temporal": {
        "description": "Past-oriented vs future-oriented language",
        "positive": [
            "Yesterday we completed the final phase of the project successfully.",
            "The ancient civilization flourished thousands of years ago in this valley.",
            "I remember when we used to play in the fields as children.",
            "The company was founded in 1923 by a small group of engineers.",
            "Last summer we traveled across Europe and visited twelve countries.",
            "The historical records show that the treaty was signed in 1648.",
            "When I was young, my grandmother used to tell me stories every night.",
            "The factory closed down five years ago and the town never recovered.",
            "We looked back at the data from previous quarters to spot the trend.",
            "The building was constructed in the Victorian era and still stands today.",
            "Three decades ago, this neighborhood was completely different.",
            "I graduated from university in 2015 and immediately started working.",
            "The tradition originated centuries ago in a small village in the mountains.",
            "They had already finished the repairs before the storm arrived.",
            "The fossil was discovered in 1987 during a routine excavation.",
            "I once visited that museum when I was traveling through the region.",
            "The kingdom fell after a prolonged siege that lasted several months.",
            "Back in the day, people relied on handwritten letters for communication.",
            "She had worked at the hospital for twenty years before retiring.",
            "The earthquake destroyed much of the city in the early morning hours.",
            "Our ancestors migrated across the continent thousands of years ago.",
            "The original manuscript was written in the fourteenth century.",
            "I recall attending my first concert when I was just sixteen years old.",
            "The river changed course after the great flood of 1993.",
            "Previously, the process required manual intervention at every step.",
            "He had been warning about the risks for years before anyone listened.",
            "The painting was commissioned by a wealthy merchant in the Renaissance.",
            "We used to gather around the radio every evening to listen to the news.",
            "The old bridge collapsed during the winter storms two years ago.",
            "Long ago, the forest stretched for hundreds of miles in every direction.",
            "The ancient library of Alexandria was one of the greatest repositories of knowledge.",
            "I distinctly remember the day the Berlin Wall fell in 1989.",
            "The Roman Empire reached its greatest extent under Emperor Trajan.",
            "My grandfather fought in the war and rarely spoke about his experiences.",
            "The original recipe was handed down through five generations of our family.",
            "They excavated the ruins and found pottery dating back to the Bronze Age.",
            "The industrial revolution transformed society in ways nobody had anticipated.",
            "I spent my childhood summers at my uncle's farm in the countryside.",
            "The peace agreement was reached after months of grueling negotiations.",
            "The old photographs captured a way of life that has long since vanished.",
            "The cathedral took over two hundred years to complete and was finished in 1880.",
            "We once had a thriving textile industry in this region before it declined.",
            "The mathematician published the proof in 1637 in the margin of a book.",
            "I vividly recall the taste of my mother's cooking from when I was small.",
            "The expedition set out in 1911 and reached the South Pole that December.",
            "The invention of the printing press in the fifteenth century changed everything.",
            "Our family emigrated from Ireland during the great famine of the 1840s.",
            "The last time this comet was visible from Earth was in 1986.",
            "The village had been abandoned for decades before archaeologists arrived.",
            "I used to walk to school every day along that narrow, winding path.",
            "The ancient oak tree in the square was planted over three hundred years ago.",
            "The space shuttle Challenger disaster in 1986 shocked the entire world.",
            "My parents met at a dance hall in 1965 and married the following year.",
            "The dynasty ruled for nearly four centuries before its eventual collapse.",
            "We harvested the last crop before the first frost arrived in October.",
            "The detective finally solved the cold case that had baffled police for years.",
            "The telegraph revolutionized long-distance communication in the nineteenth century.",
            "I remember the exact moment I learned to ride a bicycle as a child.",
            "The volcanic eruption buried the city of Pompeii in the year 79 AD.",
            "The team celebrated after winning the tournament for the third consecutive year.",
        ],
        "negative": [
            "Tomorrow we will begin the next phase of the project.",
            "In the coming decades, this technology will transform entire industries.",
            "I am looking forward to when we can explore new possibilities together.",
            "The company plans to expand into new markets by the year 2028.",
            "Next summer we will travel to Asia and visit several countries.",
            "Future historians will look back on this period as a turning point.",
            "When I am older, I hope to tell my grandchildren these stories.",
            "The new factory will open in two years and create thousands of jobs.",
            "We need to project forward and anticipate trends for the next quarter.",
            "The building will be completed by next spring according to the schedule.",
            "In three decades, this neighborhood will be completely transformed.",
            "I will graduate next year and then begin my career in research.",
            "New traditions will emerge as our culture continues to evolve.",
            "They will finish the repairs before the next storm season arrives.",
            "Future excavations will likely reveal even more significant finds.",
            "I plan to visit that museum when I travel through the region next month.",
            "The new government will implement sweeping reforms in the years ahead.",
            "Soon, people will rely on entirely new forms of communication.",
            "She will work at the hospital for many more years before considering retirement.",
            "The new safety measures will prevent such disasters from happening again.",
            "Our descendants will colonize other planets in the centuries to come.",
            "The upcoming publication will change how we think about this subject.",
            "I anticipate attending many more concerts as new artists emerge.",
            "The river will be restored through an ambitious conservation project.",
            "Going forward, the process will be fully automated with no manual steps.",
            "People will eventually recognize the importance of what we are building now.",
            "The planned monument will be the largest of its kind in the world.",
            "We will gather virtually in the future for meetings and collaboration.",
            "The proposed bridge will withstand even the most extreme weather events.",
            "One day, the forest will be replanted and stretch for miles once again.",
            "The next generation of students will have access to tools we cannot yet imagine.",
            "By 2050, renewable energy will power the vast majority of the global economy.",
            "The city will unveil a new public transit system within the next five years.",
            "My children will grow up in a world shaped by artificial intelligence.",
            "The recipe will be passed on to future generations who will adapt it further.",
            "Archaeologists will continue to uncover secrets buried beneath the surface.",
            "The coming technological revolution will dwarf anything humanity has seen before.",
            "I will spend my retirement summers traveling to places I have always dreamed of.",
            "A new peace framework will be established to prevent future conflicts.",
            "Digital archives will preserve photographs for centuries to come.",
            "The new cathedral project is expected to take twenty years to complete.",
            "Emerging markets will become the primary engines of global growth.",
            "A young mathematician will one day prove or disprove the remaining conjecture.",
            "I look forward to tasting new cuisines as I explore the world.",
            "The next Mars mission is scheduled to launch in 2030 with a human crew.",
            "Future innovations in printing will make publishing accessible to everyone.",
            "Our grandchildren will forge new connections across continents yet undeveloped.",
            "The comet will return and be visible from Earth again in 2061.",
            "The abandoned village will be transformed into an eco-tourism destination.",
            "Children will walk to school along newly built, safe pedestrian paths.",
            "The sapling planted today will grow into a towering tree for future generations.",
            "Space agencies are preparing for missions that will redefine human exploration.",
            "My children will meet their future partners in ways I cannot yet foresee.",
            "The emerging dynasty of researchers will push the boundaries of knowledge.",
            "We will harvest the benefits of this investment for decades to come.",
            "Detectives of the future will use AI tools that are still being developed.",
            "Quantum networks will revolutionize communication within the next few decades.",
            "I will teach my grandchildren to ride bicycles just as my parents taught me.",
            "Rising sea levels will reshape coastlines dramatically over the next century.",
            "The team is preparing for next year's tournament with renewed determination.",
        ],
    },
    "complexity": {
        "description": "Technical/complex language vs simple/plain language",
        "positive": [
            "The eigenvalues of the Hessian matrix indicate saddle points in the loss landscape.",
            "We employed a stochastic variational inference framework with amortized posteriors.",
            "The asymptotic complexity of the algorithm is O(n log n) in the average case.",
            "Quantum entanglement enables non-local correlations that violate Bell inequalities.",
            "The endoplasmic reticulum facilitates post-translational modification of polypeptides.",
            "Implementing a lock-free concurrent hash map requires careful memory ordering semantics.",
            "The Lagrangian formulation yields the Euler-Lagrange equations of motion for the system.",
            "Differential gene expression analysis revealed upregulation of pro-inflammatory cytokines.",
            "The compiler performs static single assignment transformation before register allocation.",
            "Topological insulators exhibit conducting surface states protected by time-reversal symmetry.",
            "We derived the posterior distribution using Markov chain Monte Carlo sampling methods.",
            "The Navier-Stokes equations govern the dynamics of viscous incompressible fluid flows.",
            "Cross-attention mechanisms compute scaled dot-product similarity between heterogeneous embeddings.",
            "The spectral decomposition of the adjacency matrix reveals community structure in the graph.",
            "Epigenetic modifications including DNA methylation regulate transcriptional silencing.",
            "We implemented a distributed consensus protocol based on the Raft algorithm.",
            "The renormalization group flow describes how coupling constants evolve with energy scale.",
            "Proteomics analysis using tandem mass spectrometry identified novel phosphorylation sites.",
            "The kernel trick implicitly maps features into a high-dimensional reproducing kernel Hilbert space.",
            "Chromatic dispersion in optical fibers causes pulse broadening proportional to fiber length.",
            "The categorical cross-entropy loss is minimized via backpropagation through the computational graph.",
            "Ribosomal frameshifting enables the translation of overlapping open reading frames.",
            "We applied a convolutional neural network with residual skip connections and batch normalization.",
            "The partition function encodes the thermodynamic properties of the canonical ensemble.",
            "Non-equilibrium phase transitions exhibit universality classes distinct from equilibrium systems.",
            "The garbage collector uses a generational tri-color marking algorithm with write barriers.",
            "Gauge symmetry breaking through the Higgs mechanism generates masses for vector bosons.",
            "We characterized the bifurcation diagram of the logistic map in the chaotic regime.",
            "The homology groups of the simplicial complex reveal topological invariants of the manifold.",
            "Allosteric regulation of enzyme kinetics follows the Monod-Wyman-Changeux concerted model.",
            "The Jacobian matrix of the transformation determines the local stretching and rotation of space.",
            "Bayesian hyperparameter optimization with Gaussian processes explores the acquisition landscape.",
            "The Kramers-Kronig relations connect the real and imaginary parts of the susceptibility.",
            "We employed a variational autoencoder with a disentangled beta-VAE objective.",
            "The Green's function propagator encodes the spectral density of the interacting many-body system.",
            "Topological data analysis uses persistent homology to extract multi-scale features.",
            "The Lindblad master equation governs the dissipative evolution of open quantum systems.",
            "We implemented gradient checkpointing to trade compute for memory in deep transformer training.",
            "The conformal field theory describes critical phenomena at second-order phase transitions.",
            "Sobolev embedding theorems establish the regularity of solutions to elliptic partial differential equations.",
            "The mitogen-activated protein kinase cascade transduces extracellular signals to transcription factors.",
            "We utilized a contrastive predictive coding objective to learn self-supervised representations.",
            "The Jones polynomial is a knot invariant computed from the Temperley-Lieb algebra.",
            "Metagenomic shotgun sequencing revealed previously uncharacterized microbial taxa in the gut.",
            "The Riemannian curvature tensor encodes the intrinsic geometry of the underlying manifold.",
            "Lock-free compare-and-swap primitives ensure linearizability in concurrent data structures.",
            "The Feynman path integral sums over all possible field configurations weighted by the action.",
            "We performed dimensionality reduction using uniform manifold approximation and projection.",
            "The chiral anomaly in quantum chromodynamics violates classical axial symmetry conservation.",
            "Differentially private stochastic gradient descent adds calibrated Gaussian noise to gradient updates.",
            "The Hodge decomposition theorem decomposes differential forms on compact Riemannian manifolds.",
            "We trained a graph neural network using message passing on molecular conformer ensembles.",
            "The Kolmogorov complexity of a string measures its algorithmic incompressibility.",
            "Spintronic devices exploit electron spin angular momentum for non-volatile memory applications.",
            "We applied tensor network methods to approximate the ground state of the quantum Hamiltonian.",
            "The Atiyah-Singer index theorem relates analytic and topological invariants of elliptic operators.",
            "CRISPR-Cas9 ribonucleoprotein complexes induce site-specific double-strand breaks for genome editing.",
            "We implemented speculative decoding to accelerate autoregressive inference in large language models.",
            "The Yang-Baxter equation ensures integrability of the two-dimensional statistical mechanical model.",
            "Persistent memory programming models require explicit cache-line flushing for crash consistency.",
        ],
        "negative": [
            "The numbers in the table show where the problems are in the results.",
            "We used a common method to guess the missing values in our data.",
            "The program runs faster when you sort the list before searching it.",
            "Tiny particles can be connected in a way that seems almost magical.",
            "A part inside the cell helps build and fold the proteins it needs.",
            "Making a program that many users can use at once is tricky.",
            "The math equation tells us how the ball moves through the air.",
            "Some genes became more active and caused swelling in the tissue.",
            "The program turns your code into a simpler form before running it.",
            "Some special materials can conduct electricity only on their surface.",
            "We used random sampling to figure out the most likely answer.",
            "The equation describes how water and other liquids flow and move.",
            "The model looks at two different inputs and figures out how similar they are.",
            "By looking at the connections, we can find groups in the network.",
            "Small chemical tags on DNA can turn genes on and off.",
            "We built a system where multiple computers agree on the same answer.",
            "The strength of a force changes depending on how much energy you use.",
            "Scientists used special tools to find which parts of proteins are active.",
            "A math trick lets us compare things in a much richer way.",
            "Light signals get stretched out as they travel through long glass cables.",
            "The model learns by adjusting its weights to reduce the error score.",
            "Sometimes cells read the genetic code in a slightly shifted way.",
            "We used a layered model that can skip ahead and learn faster.",
            "One number captures all the important thermal properties of the system.",
            "Some systems change dramatically when pushed far from their normal state.",
            "The program automatically frees up memory that is no longer being used.",
            "A special field gives other particles their mass through an interaction.",
            "The simple equation can produce wildly unpredictable behavior at certain settings.",
            "The shape of an object can be described by counting its holes and surfaces.",
            "A molecule can change shape when another molecule binds to a different spot.",
            "A grid of numbers tells you how space gets stretched and rotated nearby.",
            "The computer tries many settings and picks whichever works best.",
            "Two related measurements are always linked by a simple mathematical rule.",
            "The model learns to compress data into a small code and then rebuild it.",
            "A mathematical tool tracks how signals bounce around inside a system.",
            "We look at the shape of data at different zoom levels to find patterns.",
            "The equation describes how a system slowly leaks energy into its surroundings.",
            "The computer saves its work along the way to use less memory.",
            "At special temperatures, materials behave in ways that follow universal rules.",
            "Smooth solutions to certain equations can be guaranteed under the right conditions.",
            "A chain of chemical signals carries messages from outside the cell to the nucleus.",
            "The model learns patterns by predicting what comes next in the data.",
            "A number assigned to a knot stays the same no matter how you twist it.",
            "Scientists read all the DNA in a sample to discover new types of microbes.",
            "A mathematical object captures how curved a space is at every point.",
            "Special computer instructions let multiple threads update data without locking.",
            "The calculation adds up every possible path a particle could take.",
            "We shrank the data down to two dimensions so we could draw a picture of it.",
            "A rule from particle physics says that a certain symmetry does not quite hold.",
            "The learning algorithm adds a bit of random noise to protect personal data.",
            "Any smooth field on a compact surface can be split into three simple parts.",
            "The model passes messages between atoms in a molecule to learn its properties.",
            "The shortest possible description of a message measures how random it is.",
            "Tiny magnets inside electronics can store information without needing power.",
            "We used a network of small blocks to find the lowest energy state of a system.",
            "A deep theorem connects the number of solutions to the shape of the space.",
            "A molecular tool cuts DNA at a precise spot so scientists can edit genes.",
            "The computer guesses several words ahead to generate text more quickly.",
            "A special equation guarantees that a puzzle in physics can be solved exactly.",
            "Programs that survive power failures need to carefully save data to special memory.",
        ],
    },
    "subjectivity": {
        "description": "Subjective opinion vs objective factual statement",
        "positive": [
            "I believe this is the most beautiful painting ever created by anyone.",
            "In my opinion, Italian food is far superior to any other cuisine.",
            "The movie was absolutely wonderful, easily the best film of the year.",
            "I feel that remote work is much better than working in an office.",
            "This novel is overrated and I personally found it quite boring.",
            "I think classical music is more intellectually stimulating than pop music.",
            "From my perspective, the city is a much more exciting place to live.",
            "The new policy is a terrible idea and will only make things worse.",
            "I strongly prefer warm weather over cold, winter is just miserable.",
            "This design is ugly and lacks the elegance of the previous version.",
            "The best way to learn a language is through immersion, nothing else compares.",
            "I consider this to be an unacceptable level of service quality.",
            "The ending of the story was deeply unsatisfying and felt rushed.",
            "In my view, this candidate is clearly the most qualified for the position.",
            "Rock music from the seventies was infinitely better than today's music.",
            "I find modern architecture cold and uninviting compared to classical styles.",
            "This is by far the worst restaurant I have ever been to.",
            "The countryside is so much more peaceful and pleasant than the noisy city.",
            "I think people who wake up early are more productive, it just makes sense.",
            "The sequel was a massive disappointment compared to the brilliant original.",
            "Tea is a far more refined beverage than coffee, in my humble opinion.",
            "The presentation was incredibly dull and could have been half as long.",
            "I personally feel that handwriting is a dying art that we should preserve.",
            "This neighborhood has the best community spirit of anywhere I have lived.",
            "Autumn is clearly the most beautiful season, with its stunning colors.",
            "I am convinced that this technology will be remembered as a failure.",
            "The old system was much more user-friendly than this confusing new interface.",
            "Nothing beats a good book on a rainy afternoon, it is pure bliss.",
            "I think the education system is fundamentally broken and needs reform.",
            "This song is annoying and I cannot understand why anyone would enjoy it.",
            "In my experience, small teams consistently outperform large bureaucratic ones.",
            "The original version had so much more charm than this soulless remake.",
            "I genuinely believe that art is more important than science for human happiness.",
            "This neighborhood cafe makes the best espresso I have ever tasted in my life.",
            "Personally, I find podcasts far more engaging than traditional radio shows.",
            "The director's vision was brilliant but the execution was deeply flawed.",
            "I would argue that traveling alone is a far richer experience than group tours.",
            "This is hands down the most inspiring lecture I have ever attended.",
            "I am firmly of the opinion that print books are superior to e-readers.",
            "The new park design is soulless and completely lacks the old park's character.",
            "Home-cooked meals are infinitely more satisfying than anything from a restaurant.",
            "I think social media has done more harm than good to our society.",
            "This sunset is the most breathtaking natural spectacle I have ever witnessed.",
            "In my judgment, the prosecution's argument was far more compelling.",
            "I consider this album to be the defining masterpiece of the entire genre.",
            "The earlier model was so much more reliable, they really dropped the ball.",
            "I passionately believe that everyone should learn to play a musical instrument.",
            "This wine is exquisite, absolutely the finest I have had the pleasure of tasting.",
            "I think working from a coffee shop is infinitely better than a stuffy office.",
            "The modern renovation completely ruined what was once a charming historic building.",
            "Dogs are far better companions than cats, and nothing will change my mind.",
            "I feel strongly that summer holidays should be at least six weeks long.",
            "This documentary was profoundly moving and deserves every award it receives.",
            "In my estimation, the risks of this venture far outweigh the potential rewards.",
            "The handmade version has an authenticity that the factory product completely lacks.",
            "I am certain that history will judge this decision as a catastrophic mistake.",
            "There is no finer pleasure in life than a long walk through an ancient forest.",
            "I wholeheartedly believe that kindness is the most undervalued human quality.",
            "This new policy is misguided at best and deliberately harmful at worst.",
            "I would take a quiet evening at home over a noisy party any day of the week.",
        ],
        "negative": [
            "Water boils at one hundred degrees Celsius at standard atmospheric pressure.",
            "The Earth orbits the Sun at an average distance of about 150 million kilometers.",
            "The population of Tokyo is approximately 14 million people as of the latest census.",
            "Carbon dioxide is a molecule composed of one carbon and two oxygen atoms.",
            "The speed of light in a vacuum is approximately 299,792 kilometers per second.",
            "The Amazon River is the largest river by discharge volume in the world.",
            "Humans have 23 pairs of chromosomes in each cell of their body.",
            "Mount Everest has an elevation of 8,849 meters above sea level.",
            "The periodic table contains 118 confirmed chemical elements as of today.",
            "The Great Wall of China stretches approximately 21,196 kilometers in total length.",
            "The human heart beats approximately 100,000 times per day on average.",
            "Oxygen makes up about 21 percent of the Earth's atmosphere by volume.",
            "The Sahara Desert covers approximately 9.2 million square kilometers in Africa.",
            "An adult human body contains roughly 206 bones in total.",
            "The International Space Station orbits the Earth approximately every 90 minutes.",
            "DNA is structured as a double helix composed of nucleotide base pairs.",
            "The Nile River flows through eleven countries in northeastern Africa.",
            "Sound travels at approximately 343 meters per second through air at room temperature.",
            "The currency of Japan is the yen, issued by the Bank of Japan.",
            "Jupiter is the largest planet in our solar system by both mass and volume.",
            "Photosynthesis converts carbon dioxide and water into glucose and oxygen.",
            "The first programmable electronic computer was completed in 1945.",
            "Average global surface temperature has risen by about 1.1 degrees since 1900.",
            "The Pacific Ocean covers approximately 165 million square kilometers.",
            "The United Nations was established in 1945 with 51 founding member states.",
            "Insulin is a peptide hormone produced by beta cells in the pancreas.",
            "The Eiffel Tower stands 330 meters tall including its antenna.",
            "The Pythagorean theorem states that a squared plus b squared equals c squared.",
            "The first successful powered airplane flight occurred on December 17, 1903.",
            "Adult humans typically have 32 teeth including wisdom teeth.",
            "A standard chess board contains 64 squares arranged in an eight-by-eight grid.",
            "The chemical symbol for gold is Au, derived from the Latin word aurum.",
            "Mars has two small moons named Phobos and Deimos.",
            "Caffeine is a central nervous system stimulant found in coffee and tea.",
            "The Atlantic Ocean separates the Americas from Europe and Africa.",
            "A kilometer is equal to one thousand meters in the metric system.",
            "The human brain contains approximately 86 billion neurons.",
            "Diamonds are composed of carbon atoms arranged in a crystal lattice structure.",
            "The Celsius and Fahrenheit temperature scales intersect at minus forty degrees.",
            "Bees communicate the location of food sources through a pattern known as the waggle dance.",
            "The circumference of the Earth at the equator is approximately 40,075 kilometers.",
            "Sodium chloride, commonly known as table salt, has the chemical formula NaCl.",
            "The Mariana Trench is the deepest known point in the ocean at about 11,034 meters.",
            "Copper is an excellent conductor of both electricity and heat.",
            "The moon takes approximately 27.3 days to complete one orbit around the Earth.",
            "Helium is the second lightest and second most abundant element in the observable universe.",
            "The boiling point of ethanol is approximately 78.37 degrees Celsius.",
            "A standard year in the Gregorian calendar contains 365 days.",
            "The mitochondrion is often described as the powerhouse of the cell.",
            "The Richter scale measures the magnitude of seismic waves produced by earthquakes.",
            "Sharks have been present in Earth's oceans for over 400 million years.",
            "The distance from the Earth to the Moon is approximately 384,400 kilometers.",
            "Nitrogen constitutes about 78 percent of the Earth's atmosphere by volume.",
            "The Panama Canal connects the Atlantic and Pacific Oceans across Central America.",
            "A light-year is the distance that light travels in one Julian year.",
            "The human genome contains approximately 3 billion base pairs of DNA.",
            "Iron has an atomic number of 26 on the periodic table of elements.",
            "The Amazon rainforest produces roughly 20 percent of the world's oxygen.",
            "Mercury is the closest planet to the Sun in our solar system.",
            "An octave in music represents a doubling of the frequency of a sound wave.",
        ],
    },
    "emotion_joy_anger": {
        "description": "Joyful/warm emotion vs angry/hostile emotion",
        "positive": [
            "I burst out laughing when I saw the surprise they had prepared for me.",
            "Holding the newborn baby filled me with an indescribable warmth and tenderness.",
            "The children playing in the park reminded me of the simple joys in life.",
            "I could not stop smiling after hearing the wonderful news from my friend.",
            "The unexpected gift from a stranger brightened my entire week.",
            "Dancing in the living room with my family is my favorite thing in the world.",
            "Seeing the flowers bloom after the long winter filled me with delight.",
            "The puppy curled up in my lap and I felt completely content and at peace.",
            "Reuniting with old friends after years apart was a moment of pure bliss.",
            "The smell of fresh cookies baking brings back the happiest childhood memories.",
            "I felt a warm glow of pride watching my daughter receive her diploma.",
            "The gentle rain on the rooftop creates the most soothing and peaceful atmosphere.",
            "Sharing a meal with loved ones is one of life's greatest simple pleasures.",
            "The kind words from my colleague made me feel valued and appreciated.",
            "I was filled with wonder as I watched the northern lights for the first time.",
            "The gentle breeze carried the scent of lavender and it was absolutely lovely.",
            "I felt so fortunate to witness such a beautiful moment of human kindness.",
            "Cuddling under a blanket with hot chocolate is my idea of perfect happiness.",
            "The laughter of the crowd was infectious and soon everyone was smiling.",
            "Waking up without an alarm on a peaceful Sunday morning is pure bliss.",
            "The artist's performance was so moving it brought tears of happiness to my eyes.",
            "Finding the perfect gift for someone you love is such a satisfying feeling.",
            "The garden party was filled with music, laughter, and warm conversation.",
            "I felt a deep sense of gratitude as I watched the sunset with my partner.",
            "The child's face lit up with wonder when she saw the Christmas tree.",
            "I am overjoyed to share this wonderful news with everyone I care about.",
            "The cozy fireplace and soft music made the evening absolutely magical.",
            "A spontaneous act of kindness from a neighbor made my day infinitely better.",
            "I treasure the memory of us all laughing together around the dinner table.",
            "The world feels brighter and more beautiful when you are surrounded by love.",
            "The butterflies in my stomach turned to pure elation when I heard the good news.",
            "I giggled uncontrollably at the silly faces my nephew was making.",
            "The warmth of the campfire and the sound of guitar strings filled me with peace.",
            "Receiving flowers for no particular reason made my heart sing with happiness.",
            "I danced in the rain and felt a childlike freedom I had almost forgotten.",
            "The choir's harmonies swelled and a wave of pure bliss washed over me.",
            "I squeezed my partner's hand and felt a surge of love I could barely contain.",
            "The baby's first giggle was the most delightful sound I have ever heard.",
            "We toasted marshmallows and shared stories, and I felt completely at home.",
            "The surprise reunion with my college friends left me grinning for days.",
            "I beamed with pride as my son read his first sentence aloud to the family.",
            "The golden autumn leaves drifting down filled me with a serene kind of joy.",
            "I felt a rush of pure happiness when the crowd sang along to my song.",
            "Curling up with my cat purring on my chest is the definition of contentment.",
            "The heartfelt standing ovation brought joyful tears streaming down my cheeks.",
            "I savored every bite of the homemade pie and smiled at the memories it evoked.",
            "The soft glow of candles and the company of friends made the evening perfect.",
            "I could feel my spirits lift the moment I stepped into the sunlit garden.",
            "Watching the dolphins leap alongside the boat filled everyone with pure delight.",
            "The tender lullaby my mother used to sing still fills me with deep comfort.",
            "I threw my arms around my friend and we both laughed until we cried.",
            "The first sip of hot cocoa on a freezing day was absolute heaven.",
            "We built a blanket fort and spent the evening watching movies in pure bliss.",
            "I felt an overwhelming wave of gratitude as I looked at my sleeping children.",
            "The festival was alive with color, music, and the most infectious joyfulness.",
            "I caught a glimpse of a rainbow and it felt like a sign of good things to come.",
            "The applause after the school recital made every hour of practice worthwhile.",
            "I felt giddy with excitement as I unpacked the boxes in my very first home.",
            "The simple pleasure of warm bread straight from the oven is hard to beat.",
            "My grandmother's embrace always had a way of making every worry disappear.",
        ],
        "negative": [
            "I am absolutely furious about the way they handled this entire situation.",
            "The blatant disrespect shown by the manager made my blood boil with rage.",
            "I slammed the door shut because I could not stand being in that room anymore.",
            "How dare they make such a reckless decision without consulting anyone first!",
            "The constant lies and manipulation make me want to scream in frustration.",
            "I am seething with anger after discovering the deliberate sabotage of my work.",
            "Their arrogant dismissal of my concerns is absolutely infuriating and offensive.",
            "I clenched my fists as the injustice of the situation became painfully clear.",
            "The sheer incompetence on display is enough to make anyone lose their temper.",
            "I am fed up with the endless excuses and total lack of accountability.",
            "The betrayal cut deep and now all I feel is burning resentment.",
            "I wanted to throw the phone across the room after that enraging conversation.",
            "The bureaucratic runaround has pushed me to the absolute limit of my patience.",
            "Their smug attitude while everyone else suffers is utterly contemptible.",
            "I cannot believe the audacity of someone who would do something so callous.",
            "The unfair treatment of my colleagues has left me absolutely livid.",
            "Every time they interrupt me I feel a surge of barely contained rage.",
            "The vandalism of the community garden was a senseless act that enraged everyone.",
            "I am outraged by the corruption and greed that caused this preventable disaster.",
            "Their complete disregard for the rules while others follow them is maddening.",
            "I stormed out of the meeting because I refused to tolerate any more nonsense.",
            "The hateful comments online filled me with a burning desire to fight back.",
            "I am incandescent with fury at the injustice that has been allowed to continue.",
            "The deliberate cruelty shown towards the vulnerable is absolutely unforgivable.",
            "I pounded the table in frustration as yet another promise was broken.",
            "Their pathetic attempt at an apology only made me angrier than before.",
            "I have never been so enraged in my life as I was at that moment.",
            "The reckless negligence that caused the accident makes me shake with anger.",
            "I am boiling with indignation at the way they twisted the truth.",
            "The hostile and aggressive tone of the letter left me feeling attacked and furious.",
            "I was so angry I could barely see straight as I read the verdict.",
            "The brazen theft of credit for my work made my blood pressure skyrocket.",
            "I gritted my teeth as they dismissed months of effort with a casual shrug.",
            "The toxic culture in this office is enough to make anyone snap with rage.",
            "I wanted to scream when the same mistake happened for the fifth time in a row.",
            "Their condescending tone made me want to slam my laptop shut and walk away.",
            "I am livid that they knowingly endangered people's lives to save a few dollars.",
            "The utter contempt in their voice was enough to make my hands tremble with fury.",
            "I have never felt such white-hot rage as when I discovered the deception.",
            "The senseless destruction of the historic building left the community seething.",
            "I threw down the report in disgust after reading the blatant distortions inside.",
            "Their gleeful mockery of those less fortunate fills me with righteous anger.",
            "I am absolutely enraged that they broke every promise they made to our faces.",
            "The cavalier attitude towards safety violations makes my stomach churn with fury.",
            "I kicked the chair across the room after reading the insulting response.",
            "Their refusal to take any responsibility whatsoever is driving me up the wall.",
            "I was shaking with anger as I listened to the recording of the meeting.",
            "The cruel joke at my expense in front of everyone made me burn with humiliation.",
            "I slammed my fist on the desk because I could not tolerate one more excuse.",
            "The deliberate cover-up of the truth has ignited a firestorm of public outrage.",
            "I ground my teeth in fury as they casually blamed everyone except themselves.",
            "Their sneering dismissal of genuine concerns is the most infuriating thing imaginable.",
            "I am apoplectic with rage at the sheer scale of the injustice committed here.",
            "The bully's taunting laughter echoed in my ears and I could feel my rage building.",
            "I yanked the cord from the wall in frustration after the third crash in an hour.",
            "Their willful ignorance in the face of overwhelming evidence is maddening beyond belief.",
            "I felt a surge of fury so intense it took every ounce of restraint not to shout.",
            "The fact that no one has been held accountable makes my anger burn even hotter.",
            "I was fuming silently in my seat as they took credit for the entire project.",
            "The vicious personal attack in the email left me trembling with anger and disbelief.",
        ],
    },
    "instruction": {
        "description": "Imperative instructional text vs descriptive narrative text",
        "positive": [
            "First, preheat the oven to 350 degrees and grease a baking pan.",
            "Open the terminal and type the following command to install the package.",
            "Remove the four screws from the back panel using a Phillips screwdriver.",
            "Add two cups of flour to the bowl and mix until a smooth dough forms.",
            "Navigate to the settings menu and click on the privacy tab.",
            "Insert the battery into the compartment with the positive terminal facing up.",
            "Stir the mixture continuously over medium heat until it thickens.",
            "Download the application from the official website and run the installer.",
            "Turn off the power supply before disconnecting any of the cables.",
            "Fold the paper in half along the dotted line and crease it firmly.",
            "Apply a thin layer of adhesive to both surfaces and press them together.",
            "Enter your username and password in the fields provided, then click submit.",
            "Sand the surface with fine grit sandpaper until it is smooth to the touch.",
            "Connect the red wire to the positive terminal and the black to the negative.",
            "Place the seedlings two inches apart in rows and water them immediately.",
            "Save your work frequently to avoid losing any changes during the process.",
            "Measure the length of the wall and cut the board to the correct size.",
            "Hold the stretch for thirty seconds, then slowly release and repeat.",
            "Shake the bottle thoroughly before opening and pour the recommended dose.",
            "Attach the bracket to the wall using the provided anchors and screws.",
            "Run the diagnostic tool and note any error codes that appear on screen.",
            "Rinse the filter under running water and let it dry before reinstalling it.",
            "Select all the files you wish to transfer and drag them to the folder.",
            "Tighten the bolts in a star pattern to ensure even pressure distribution.",
            "Set the timer for fifteen minutes and do not open the lid until it rings.",
            "Click the export button and choose the format you would like to save in.",
            "Apply even pressure while cutting along the marked line with the utility knife.",
            "Disconnect the old component and replace it with the new one provided.",
            "Align the edges carefully before stapling the pages together.",
            "Follow the on-screen prompts to complete the registration process.",
            "Drain the pasta when it is al dente and immediately toss it with the sauce.",
            "Press and hold the power button for five seconds to perform a hard reset.",
            "Thread the needle with eighteen inches of thread and tie a knot at the end.",
            "Rotate the dial clockwise until you hear a click, then pull the handle.",
            "Copy the configuration file to the backup directory before making changes.",
            "Spread the mortar evenly with a trowel and lay each tile firmly into place.",
            "Unplug the appliance and allow it to cool completely before cleaning.",
            "Highlight the text you wish to format and select bold from the toolbar.",
            "Pour the batter into the prepared pan and smooth the top with a spatula.",
            "Check the oil level using the dipstick and add more if it is below the mark.",
            "Create a new branch in the repository before committing your changes.",
            "Inflate the tire to the recommended pressure shown on the sidewall.",
            "Dissolve the tablet in a full glass of water and drink it immediately.",
            "Secure the harness by pulling the strap until it fits snugly across your chest.",
            "Back up your database before running the migration script.",
            "Prime the pump by pressing the button three times before the first use.",
            "Trim the excess fabric with sharp scissors and fold the hem under.",
            "Restart the router by unplugging it for thirty seconds, then plugging it back in.",
            "Label each sample clearly with the date, time, and identification number.",
            "Calibrate the instrument by following the steps outlined in the user manual.",
            "Clean the wound gently with saline solution and cover it with a sterile bandage.",
            "Mount the shelf brackets at exactly the same height using a spirit level.",
            "Compile the source code with optimization flags enabled for best performance.",
            "Blanch the vegetables in boiling water for two minutes, then plunge into ice water.",
            "Verify that all connections are secure before turning on the main power switch.",
            "Write your name and date of birth in block capitals on the first line.",
            "Flatten the dough with a rolling pin until it is about a quarter inch thick.",
            "Synchronize your device by connecting it to the computer and clicking sync.",
            "Lubricate the chain with a few drops of oil and wipe away any excess.",
            "Submit the completed form to the front desk before the end of business today.",
        ],
        "negative": [
            "The oven had been preheated and the kitchen was filled with a warm aroma.",
            "The terminal window displayed a blinking cursor against a dark background.",
            "The old panel had four rusted screws that had not been touched in years.",
            "The flour created a small cloud of white dust as it fell into the bowl.",
            "The settings menu contained a long list of options organized by category.",
            "The battery compartment was located on the underside of the device.",
            "The mixture gradually thickened as the heat worked its way through evenly.",
            "The application had been downloaded millions of times since its release.",
            "The power supply hummed quietly in the corner of the server room.",
            "The paper was folded neatly with sharp creases along its edges.",
            "The adhesive created a strong bond that held the materials firmly in place.",
            "The login page featured a simple design with two input fields and a button.",
            "The surface had been sanded smooth and was ready for a coat of paint.",
            "The wires ran along the wall in neat parallel lines, red beside black.",
            "The seedlings stood in perfect rows across the freshly tilled garden bed.",
            "The document had been saved multiple times throughout the editing process.",
            "The board had been cut to the exact length needed for the shelf.",
            "She held the stretch and felt the tension gradually release from her muscles.",
            "The bottle contained a thick liquid that needed to be shaken before use.",
            "The bracket hung securely on the wall, supporting the weight of the shelf.",
            "The diagnostic report listed several warnings but no critical errors.",
            "The filter was clean and dry, sitting on the counter next to the machine.",
            "The files had been transferred to the new folder earlier that morning.",
            "The bolts were tightened evenly and the assembly felt solid and secure.",
            "The timer ticked quietly on the counter as steam rose from beneath the lid.",
            "The exported file appeared in the downloads folder in the selected format.",
            "The cut was clean and straight, following the marked line perfectly.",
            "The old component sat on the workbench next to its brand new replacement.",
            "The pages were neatly aligned and bound together with a single staple.",
            "The registration process took approximately five minutes from start to finish.",
            "The pasta had been cooked to perfection and glistened with a light coating of sauce.",
            "The power button glowed faintly in the dim light of the server room.",
            "The needle trailed a long thread that looped gracefully across the fabric.",
            "The dial had been turned to its maximum position and the handle was extended.",
            "The configuration file sat in the backup directory alongside older versions.",
            "The mortar had dried overnight and the tiles were firmly set into the floor.",
            "The appliance had cooled down and was sitting unplugged on the counter.",
            "The selected text appeared in bold across the center of the slide.",
            "The batter spread smoothly across the pan in a thin, even layer.",
            "The oil level registered just above the minimum mark on the dipstick.",
            "The new branch contained all the recent commits from the development team.",
            "The tire pressure gauge showed a reading slightly below the recommended level.",
            "The tablet dissolved quickly, turning the water a pale cloudy white.",
            "The harness fit snugly across his chest, its buckle gleaming in the sunlight.",
            "The database backup completed successfully in under three minutes.",
            "The pump primed quickly and water began to flow steadily through the pipe.",
            "The excess fabric had been trimmed neatly and the hem was perfectly even.",
            "The router blinked steadily as it re-established the network connection.",
            "Each sample was labeled with a date, time, and a unique identification number.",
            "The instrument had been recently calibrated and was reading within tolerance.",
            "The wound had been cleaned and dressed with a sterile white bandage.",
            "The shelf brackets were mounted at precisely the same height on both sides.",
            "The source code compiled without errors and the binary was ready to deploy.",
            "The vegetables had been blanched and were sitting in a bowl of ice water.",
            "All the connections had been inspected and the main power switch was turned on.",
            "His name and date of birth were printed in neat block capitals on the form.",
            "The dough had been rolled flat and lay in a pale circle on the floured surface.",
            "The device was connected and the synchronization progress bar crept forward.",
            "The chain gleamed with a thin film of fresh oil after being lubricated.",
            "The completed form sat on the front desk waiting to be collected.",
        ],
    },
}

# Number of concepts
NUM_CONCEPTS = len(CONCEPTS)

# ---------------------------------------------------------------------------
# Utility functions (used by steer.py)
# ---------------------------------------------------------------------------

def load_concept_prompts(path=PROMPTS_PATH):
    """Load saved concept prompts from disk."""
    with open(path, "r") as f:
        return json.load(f)


def load_cached_activations(concept_name, direction, layer_idx=None, position="last"):
    """
    Load cached activation arrays for a given concept and direction.

    Args:
        concept_name: e.g. "sentiment"
        direction: "positive" or "negative"
        layer_idx: if None, returns dict {layer_idx: array}, else returns single array
        position: "last" for last-token, "mean" for mean-pooled

    Returns:
        numpy array of shape (num_prompts, hidden_size) or dict of such arrays
    """
    pos_suffix = f"_{position}" if position != "last" else ""
    concept_dir = ACTIVATIONS_DIR / concept_name / direction
    if layer_idx is not None:
        path = concept_dir / f"layer_{layer_idx:02d}{pos_suffix}.npy"
        return np.load(path)
    # Load all layers
    result = {}
    for f in sorted(concept_dir.glob(f"layer_*{pos_suffix}.npy")):
        stem = f.stem.replace(pos_suffix, "")
        idx = int(stem.split("_")[1])
        result[idx] = np.load(f)
    return result


def load_all_activations(position="last"):
    """
    Load all cached activations into a nested dict.

    Returns:
        dict[concept_name][direction][layer_idx] = np.array of shape (n_prompts, hidden_size)
    """
    all_acts = {}
    for concept_name in CONCEPTS:
        all_acts[concept_name] = {}
        for direction in ["positive", "negative"]:
            all_acts[concept_name][direction] = load_cached_activations(
                concept_name, direction, position=position
            )
    return all_acts


def get_extraction_meta():
    """Load metadata from activation extraction (model info, layer count, etc.)."""
    with open(META_PATH, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# One-time setup: generate and save prompts
# ---------------------------------------------------------------------------

def generate_prompts():
    """Save concept prompts to disk as JSON."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    prompts_data = {}
    for name, concept in CONCEPTS.items():
        prompts_data[name] = {
            "description": concept["description"],
            "positive": concept["positive"][:PROMPTS_PER_DIRECTION],
            "negative": concept["negative"][:PROMPTS_PER_DIRECTION],
        }
    with open(PROMPTS_PATH, "w") as f:
        json.dump(prompts_data, f, indent=2)
    total = sum(
        len(v["positive"]) + len(v["negative"]) for v in prompts_data.values()
    )
    print(f"Saved {total} prompts across {len(prompts_data)} concepts to {PROMPTS_PATH}")
    return prompts_data


# ---------------------------------------------------------------------------
# One-time setup: extract and cache activations
# ---------------------------------------------------------------------------

def extract_activations():
    """
    Load the model, run all concept prompts through it, capture residual-stream
    activations at every layer (last token + mean pool), and save to disk.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="cpu",
        output_hidden_states=True,
    )
    model.eval()

    # Get model info
    config = model.config
    num_layers = config.num_hidden_layers
    hidden_size = config.hidden_size
    print(f"Model: {num_layers} layers, hidden_size={hidden_size}")

    # Load prompts
    prompts_data = load_concept_prompts()

    # Extract activations for each concept/direction
    ACTIVATIONS_DIR.mkdir(parents=True, exist_ok=True)
    total_prompts = 0

    for concept_name, concept_data in prompts_data.items():
        for direction in ["positive", "negative"]:
            prompts = concept_data[direction]
            save_dir = ACTIVATIONS_DIR / concept_name / direction
            save_dir.mkdir(parents=True, exist_ok=True)

            # Collect activations per layer (both last-token and mean-pool)
            layer_acts_last = {i: [] for i in range(num_layers)}
            layer_acts_mean = {i: [] for i in range(num_layers)}

            for prompt_idx, prompt in enumerate(prompts):
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    max_length=MAX_SEQ_LEN,
                    truncation=True,
                    padding=False,
                )
                with torch.no_grad():
                    outputs = model(**inputs)

                hidden_states = outputs.hidden_states
                seq_len = inputs["input_ids"].shape[1]
                last_pos = seq_len - 1

                for layer_idx in range(num_layers):
                    hs = hidden_states[layer_idx + 1][0]  # (seq_len, hidden_size)
                    # Last token
                    layer_acts_last[layer_idx].append(hs[last_pos, :].numpy())
                    # Mean pool over all tokens
                    layer_acts_mean[layer_idx].append(hs.mean(dim=0).numpy())

                if (prompt_idx + 1) % 10 == 0:
                    print(f"  [{concept_name}/{direction}] {prompt_idx + 1}/{len(prompts)}")

            # Save each layer's activations (both positions)
            for layer_idx in range(num_layers):
                arr_last = np.stack(layer_acts_last[layer_idx], axis=0)
                arr_mean = np.stack(layer_acts_mean[layer_idx], axis=0)
                np.save(save_dir / f"layer_{layer_idx:02d}.npy", arr_last)
                np.save(save_dir / f"layer_{layer_idx:02d}_mean.npy", arr_mean)

            total_prompts += len(prompts)
            print(f"  {concept_name}/{direction}: {len(prompts)} prompts, "
                  f"saved {num_layers} layers (last + mean)")

    # Save metadata
    meta = {
        "model_name": MODEL_NAME,
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "max_seq_len": MAX_SEQ_LEN,
        "num_concepts": len(prompts_data),
        "concept_names": list(prompts_data.keys()),
        "prompts_per_direction": PROMPTS_PER_DIRECTION,
        "total_prompts": total_prompts,
        "extraction_positions": EXTRACTION_POSITIONS,
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone! Extracted activations for {total_prompts} prompts across {num_layers} layers.")
    print(f"Positions: {EXTRACTION_POSITIONS}")
    print(f"Cached to: {ACTIVATIONS_DIR}")
    print(f"Metadata: {META_PATH}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-steer v2: preparation script")
    parser.add_argument("--prompts-only", action="store_true",
                        help="Only regenerate concept prompts (skip extraction)")
    parser.add_argument("--extract-only", action="store_true",
                        help="Only re-extract activations (assumes prompts exist)")
    args = parser.parse_args()

    t0 = time.time()

    if args.extract_only:
        extract_activations()
    elif args.prompts_only:
        generate_prompts()
    else:
        generate_prompts()
        extract_activations()

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
