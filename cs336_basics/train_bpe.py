import os
from typing import List
from cs336_basics.pretokenization import parallelize_pretokenization
from cs336_basics.utils import Node, DoublyLinkedList, IndexedMaxHeap

class BPEtokenizer:
    def __init__(self, special_token: List[str], vocab_size: int):
        self.vocab = {}
        self.reverse_vocab = {}
        self.vocab_size = vocab_size
        self.merges = []

        for i in range(256):
            token = bytes([i])
            self.vocab[i] = token
            self.reverse_vocab[token] = i

        for i in range(len(special_token)):
            token = special_token[i].encode("utf-8")
            self.vocab[i+256] = token
            self.reverse_vocab[token] = i+256

    
    def update(self, token: bytes):
        next_token_ID = len(self.vocab)
        self.vocab[next_token_ID] = token
        self.reverse_vocab[token] = next_token_ID


def train_bpe(
        Tokenizer: BPEtokenizer,
        input_path: str | os.PathLike,
        num_process: int,
):
   special_tokens = [Tokenizer.vocab[i].decode("utf-8") for i in range(256, Tokenizer.vocab_size) if i in Tokenizer.vocab]
   total_count = parallelize_pretokenization(input_path, num_process, special_tokens)
   vocab_chains = []

   StatisticsHeap = IndexedMaxHeap()
   PositionDictionory = {}

   for word_bytes, freq in total_count.items():
       nodes = [Node(token, freq) for token in word_bytes]

       for i in range(len(nodes)):
           if i > 0:
               nodes[i].prev = nodes[i-1]
           if i < len(nodes)-1:
               nodes[i].next = nodes[i+1]
        
       vocab_chains.append(nodes[0])

       for i in range(len(nodes)-1):
           pair = (nodes[i].value, nodes[i+1].value)

           StatisticsHeap.push(pair, freq)

           if pair not in PositionDictionory:
               PositionDictionory[pair] = []
           PositionDictionory[pair].append(nodes[i])
       
   tokenizer = Tokenizer
   invalidated_nodes = set()

   def merge():
       pair, freq = StatisticsHeap.pop()
       nodes = PositionDictionory[pair]
       tokenizer.merges.append(pair)

       for node in nodes:
           if node in invalidated_nodes:
               continue
           if node.value != pair[0] or node.next is None or node.next.value != pair[1]:
               continue
           
           node_freq = node.freq
           if node.prev:
               old_left_pair = (node.prev.value, node.value)
               StatisticsHeap.push(old_left_pair, -node_freq)
           if node.next and node.next.next:
               old_right_pair = (node.next.value, node.next.next.value)
               StatisticsHeap.push(old_right_pair, -node_freq)
           
           if node.next:
               new_token = node.value + node.next.value
               node.value = new_token
               node_to_remove = node.next
               invalidated_nodes.add(node_to_remove)
               if node_to_remove.next:
                   node_to_remove.next.prev = node
                   node.next = node_to_remove.next
               else:
                   node.next = None
               
           
           if node.prev:
               new_left_pair = (node.prev.value, node.value)
               StatisticsHeap.push(new_left_pair, node_freq)
               if new_left_pair not in PositionDictionory: PositionDictionory[new_left_pair] = []
               PositionDictionory[new_left_pair].append(node.prev)
           if node.next:
               new_right_pair = (node.value, node.next.value)
               StatisticsHeap.push(new_right_pair, node_freq)
               if new_right_pair not in PositionDictionory: PositionDictionory[new_right_pair] = []
               PositionDictionory[new_right_pair].append(node)

       del PositionDictionory[pair]


       tokenizer.update(pair[0]+pair[1])

   merge_num = tokenizer.vocab_size - len(tokenizer.vocab)

   for _ in range(merge_num):
       merge()
           