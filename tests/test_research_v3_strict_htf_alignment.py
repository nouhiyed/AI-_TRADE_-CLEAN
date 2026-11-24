from scripts.research_v3_strict_htf_alignment import aligned_values

def test_long_requires_full_bullish_stack():
    assert aligned_values({"close":110,"EMA_21":105,"EMA_50":100},"LONG")
    assert not aligned_values({"close":110,"EMA_21":99,"EMA_50":100},"LONG")
    assert not aligned_values({"close":99,"EMA_21":105,"EMA_50":100},"LONG")

def test_short_requires_full_bearish_stack():
    assert aligned_values({"close":90,"EMA_21":95,"EMA_50":100},"SHORT")
    assert not aligned_values({"close":90,"EMA_21":101,"EMA_50":100},"SHORT")

def test_missing_values_and_invalid_direction_fail_closed():
    assert not aligned_values({"close":90,"EMA_21":None,"EMA_50":100},"SHORT")
    assert not aligned_values({"close":90,"EMA_21":95,"EMA_50":100},"WAIT")
