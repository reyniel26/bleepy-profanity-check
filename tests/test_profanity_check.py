"""Test Cases"""
from profanity_check import predict, predict_prob


def test_accuracy():
    """Test Accuracy"""
    texts = [
      'Hello there, how are you',
      'Lorem Ipsum is simply dummy text of the printing and typesetting industry.',
      '!!!! Click this now!!! -> https://example.com',
      'fuck you',
      'fUcK u',
      'GO TO hElL, you dirty scum',
    ]

    assert list(predict(texts)) == [0, 1, 0, 1, 1, 1]

    probs = list(predict_prob(texts))
    for prob in probs:
        if probs.index(prob) in (0,2):
            assert prob <= 0.5
        else:
            assert prob >= 0.5


def test_edge_cases():
    """Test Edge cases"""
    texts = [
      '',
      '                    ',
      'this is but a test string, there is no offensive language to be found here! :) ' * 25,
      'aaaaaaa' * 100,
    ]
    assert list(predict(texts)) == [0, 0, 0, 0]

def test_tagalog_words():
    """Test Tagalog Words"""

    words = [
      "tanga","mahal","bobo","ikaw","talaga"
    ]
    assert list(predict(words,lang="tagalog")==[1,0,1,0,0])
