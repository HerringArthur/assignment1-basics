import regex as re
from typing import Callable 

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def encode_str_to_utf8_bytes(string: str) -> bytes:
    return string.encode("utf-8")

def decode_utf8_bytes_to_str(bytestring: bytes) -> str:
    string = []
    i = 0
    n = len(bytestring)

    while i < n:
        b = bytestring[i]

        if b // 16 == 15:        
            chunk = bytestring[i:i+4]
            string.append(chunk.decode("utf-8"))
            i += 4
        elif b // 16 == 14:      
            chunk = bytestring[i:i+3]
            string.append(chunk.decode("utf-8"))
            i += 3
        elif b // 32 == 6:       
            chunk = bytestring[i:i+2]
            string.append(chunk.decode("utf-8"))
            i += 2
        else:                    
            string.append(bytes([b]).decode("utf-8"))
            i += 1

    return "".join(string)


class BPEtokenizer:
    def __init__(self):
        for i in range(256):
            self.vocabulary[bytes([i])] = i
        
    def update(self, tokens: list[str]):
        for token in tokens:
            token = tuple(bytes([x]) for x in encode_str_to_utf8_bytes(token))
            if token in self.frequency_table:
                self.frequency_table[token] += 1
            else:
                self.frequency_table[token] = 1