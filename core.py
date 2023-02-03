from enum import Enum

class Normalization(str, Enum):
    Neg1To1 = 'neg1_to_1'
    Legacy = 'legacy'