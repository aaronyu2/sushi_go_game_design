from enum import Enum, auto

class SushiCardType(str, Enum):
    TEMPURA = 'TEMPURA'
    SASHIMI = 'SASHIMI'
    DUMPLING = 'DUMPLING'
    MAKI_ROLLS_1 = 'MAKI_ROLLS_1'
    MAKI_ROLLS_2 = 'MAKI_ROLLS_2'
    MAKI_ROLLS_3 = 'MAKI_ROLLS_3'
    SALMON_NIGIRI = 'SALMON_NIGIRI'
    SQUID_NIGIRI = 'SQUID_NIGIRI'
    EGG_NIGIRI = 'EGG_NIGIRI'
    PUDDING = 'PUDDING'
    WASABI = 'WASABI'
    SALMON_WASABI = 'SALMON_WASABI'
    SQUID_WASABI = 'SQUID_WASABI'
    EGG_WASABI = 'EGG_WASABI'
    CHOPSTICKS = 'CHOPSTICKS'
    USED_CHOPSTICKS = 'USED_CHOPSTICKS'
    HIDDEN = 'HIDDEN'


NIGIRI_VALS = {
    SushiCardType.EGG_NIGIRI: 1,
    SushiCardType.SALMON_NIGIRI: 2,
    SushiCardType.SQUID_NIGIRI: 3,
    SushiCardType.EGG_WASABI: 3,
    SushiCardType.SALMON_WASABI: 6,
    SushiCardType.SQUID_WASABI: 9,
}

PUDDING_SCORE = 6
MAKI_SCORE = [6, 3]
SASHIMI_SCORE = 10
TEMPURA_SCORE = 5