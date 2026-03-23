"""
Auto-steer: preparation and utility script.

One-time setup: load a pretrained LLM, generate contrastive concept prompts,
run forward passes capturing residual-stream activations at every layer,
and cache everything to disk as numpy arrays.

The cached activations are then consumed by steer.py (the file the agent
iterates on) for interpretability analysis.

Usage:
    uv run prepare_steer.py                # Full setup (prompts + extract)
    uv run prepare_steer.py --prompts-only # Only regenerate concept prompts
    uv run prepare_steer.py --extract-only # Only re-extract activations
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

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
MAX_SEQ_LEN = 128          # tokens per prompt (enough for concept elicitation)
TIME_BUDGET = 300           # 5 minutes per steer.py iteration
PROMPTS_PER_DIRECTION = 30  # prompts per positive/negative side per concept

CACHE_DIR = Path.home() / ".cache" / "autosteer"
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


def load_cached_activations(concept_name, direction, layer_idx=None):
    """
    Load cached activation arrays for a given concept and direction.

    Args:
        concept_name: e.g. "sentiment"
        direction: "positive" or "negative"
        layer_idx: if None, returns dict {layer_idx: array}, else returns single array

    Returns:
        numpy array of shape (num_prompts, hidden_size) or dict of such arrays
    """
    concept_dir = ACTIVATIONS_DIR / concept_name / direction
    if layer_idx is not None:
        path = concept_dir / f"layer_{layer_idx:02d}.npy"
        return np.load(path)
    # Load all layers
    result = {}
    for f in sorted(concept_dir.glob("layer_*.npy")):
        idx = int(f.stem.split("_")[1])
        result[idx] = np.load(f)
    return result


def load_all_activations():
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
                concept_name, direction
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
    activations at every layer (last token position), and save to disk.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
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

            # Collect last-token activations per layer
            layer_activations = {i: [] for i in range(num_layers)}

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

                # outputs.hidden_states: tuple of (num_layers+1) tensors
                # shape: (1, seq_len, hidden_size)
                # [0] = embedding output, [1..num_layers] = after each transformer block
                hidden_states = outputs.hidden_states

                # Use the last non-padding token position
                seq_len = inputs["input_ids"].shape[1]
                last_pos = seq_len - 1

                for layer_idx in range(num_layers):
                    # hidden_states[layer_idx + 1] = output of layer layer_idx
                    act = hidden_states[layer_idx + 1][0, last_pos, :].numpy()
                    layer_activations[layer_idx].append(act)

                if (prompt_idx + 1) % 10 == 0:
                    print(f"  [{concept_name}/{direction}] {prompt_idx + 1}/{len(prompts)}")

            # Save each layer's activations
            for layer_idx, acts in layer_activations.items():
                arr = np.stack(acts, axis=0)  # (n_prompts, hidden_size)
                np.save(save_dir / f"layer_{layer_idx:02d}.npy", arr)

            total_prompts += len(prompts)
            print(f"  {concept_name}/{direction}: {len(prompts)} prompts, saved {num_layers} layers")

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
        "extraction_position": "last_token",
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone! Extracted activations for {total_prompts} prompts across {num_layers} layers.")
    print(f"Cached to: {ACTIVATIONS_DIR}")
    print(f"Metadata: {META_PATH}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-steer: preparation script")
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
