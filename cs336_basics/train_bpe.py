import os
import json
import time
import tracemalloc
import logging
from typing import List
from tqdm import tqdm
from datetime import datetime
from cs336_basics.pretokenization import parallelize_pretokenization
from cs336_basics.utils import Node, DoublyLinkedList, IndexedMaxHeap

def setup_logging(log_file="training.log"):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

logger = logging.getLogger(__name__)


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


    def save(self, file_path:str):
        data = {
            "vocab_size": self.vocab_size,
            "vocab": {str(idx): token.decode("latin-1") for idx, token in self.vocab.items()},
            "merges": [(pair[0].decode("latin-1"), pair[1].decode("latin-1")) for pair in self.merges]
        }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Model saved to {file_path}")

   
    @classmethod
    def load(cls, flie_path:str):
        with open(flie_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        tokenizer = cls(special_token=[], vocab_size=data["vocab_size"])
        
        tokenizer.vocab = {}
        tokenizer.reverse_vocab = {}
        for idx_str, token_str in data["vocab"].items():
            idx = int(idx_str)
            token_bytes = token_str.encode("latin-1") 
            tokenizer.vocab[idx] = token_bytes
            tokenizer.reverse_vocab[token_bytes] = idx
            
        tokenizer.merges = []
        for p in data["merges"]:
            pair = (p[0].encode("latin-1"), p[1].encode("latin-1"))
            tokenizer.merges.append(pair)
            
        return tokenizer

def train_bpe(
        Tokenizer: BPEtokenizer,
        input_path: str | os.PathLike,
        num_process: int,
):
   special_tokens = [Tokenizer.vocab[i].decode("utf-8") for i in range(256, Tokenizer.vocab_size) if i in Tokenizer.vocab]
   
   logger.info("Starting Pre-tokenization (Counting words)...")
   total_count = parallelize_pretokenization(input_path, num_process, special_tokens)
   logger.info(f"Pre-tokenization complete. Found {len(total_count)} unique word fragments.")
   
   vocab_chains = []

   StatisticsHeap = IndexedMaxHeap()
   PositionDictionory = {}

   logger.info("Building initial chains and statistics...")

   for word_bytes, freq in tqdm(total_count.items(), desc="Building Chains:"):
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

   logger.info("Starting Merge loop...")
   merge_num = tokenizer.vocab_size - len(tokenizer.vocab)

   with tqdm(total=merge_num, desc="Merging Pairs", unit="pair") as pbar:
        while len(tokenizer.vocab) < tokenizer.vocab_size:
            success = merge()
            if not success:
                logger.warning("Heap is empty, cannot merge further.")
                break
            
            pbar.update(1)

   logger.info(f"Training done. Final vocab size: {len(tokenizer.vocab)}")
           

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train BPE Tokenizer")
    parser.add_argument("--input_path", type=str, required=True, help="Path to input text file")
    parser.add_argument("--save_path", type=str, help="Path to the save file")
    parser.add_argument("--vocab_size", type=int, default=10000, help="Maximum Vocabulary size")
    parser.add_argument("--workers", type=int, default=4, help="number of processes")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    base_name = os.path.splitext(args.save_path)[0]
    log_filename = f"{base_name}_{timestamp}.log"
    logger = setup_logging(log_filename)
    
    logger.info(f"Log file will be saved to: {log_filename}")

    special_tokens = ["<|endoftext|>"]
    tokenizer = BPEtokenizer(special_token=special_tokens, vocab_size=args.vocab_size)
    print(f"Start training BPE with vocab size {args.vocab_size} on {args.input_path}...")

    start_time = time.time()
    tracemalloc.start()

    try:
        train_bpe(tokenizer, args.input_path, args.workers)
        
    except KeyboardInterrupt:
        print("\n[!] Training interrupted by user.")
        exit(1)
    except Exception as e:
        print(f"\n[!] Error during training: {e}")
        exit(1)

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    
    tracemalloc.stop()
    end_time = time.time()
    
    duration_seconds = end_time - start_time
    duration_minutes = duration_seconds / 60
    peak_mem_mb = peak_mem / (1024 * 1024)  
    peak_mem_gb = peak_mem_mb / 1024        

    print("-" * 40)
    print("Training Complete!")
    print(f"Final Vocab Size: {len(tokenizer.vocab)}")
    print("-" * 40)
    print(f"Time Taken       : {duration_seconds:.2f} seconds ({duration_minutes:.2f} minutes)")
    print(f"Peak Memory Usage: {peak_mem_mb:.2f} MB ({peak_mem_gb:.4f} GB)")
    print("-" * 40)

    tokenizer.save(args.save_path)