# test_vader.py
# A quick test to confirm VADER is installed and working correctly.
# We run it on 4 sample sentences and print the sentiment scores.

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Create the VADER analyzer — this is the "engine" that scores text
analyzer = SentimentIntensityAnalyzer()

# A few test sentences with clearly different sentiments
test_sentences = [
    "The new policy was an excellent step toward economic growth.",
    "The decision faced widespread criticism and triggered violent protests.",
    "The election was held in April 2024.",
    "The leader delivered a powerful and inspiring speech that moved millions.",
]

# Run VADER on each sentence and print the scores
print("=" * 70)
print("  VADER TEST RUN")
print("=" * 70)

for sentence in test_sentences:
    scores = analyzer.polarity_scores(sentence)
    print()
    print(f"Sentence: {sentence}")
    print(f"  Positive: {scores['pos']:.3f}")
    print(f"  Negative: {scores['neg']:.3f}")
    print(f"  Neutral:  {scores['neu']:.3f}")
    print(f"  Compound: {scores['compound']:.3f}  <-- main score (-1 to +1)")

print()
print("=" * 70)
print("  TEST COMPLETE")
print("=" * 70)