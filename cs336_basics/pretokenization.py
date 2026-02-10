import os
import queue
import regex as re
import multiprocessing as mp
from typing import BinaryIO, List, Tuple
from collections import Counter

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
pretokenize = re.compile(PAT)

BYTES_CACHE = {i: bytes([i]) for i in range(256)}

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> List[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


"""
## Usage
with open(..., "rb") as f:
    num_processes = 4
    boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    # The following is a serial implementation, but you can parallelize this
    # by sending each start/end pair to a set of processes.
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        # Run pre-tokenization on your chunk and store the counts for each pre-token
"""



def split_text(text: str, special_tokens: List[str]) -> List[str]:
    if not special_tokens:
        return [text]
    
    pattern = "|".join(re.escape(token) for token in special_tokens)
    segments = re.split(pattern, text)
    
    return [s for s in segments if s]



def pretokenize_by_chunk(
        input_path: str | os.PathLike,
        start: int,
        end: int,
        special_tokens: List[str],
        q
):
    """
    pre-tokenize for each chunk and store the counts for each pre-token
    """
    counter = Counter()

    with open(input_path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="ignore")
        segments = split_text(text, special_tokens)
        
        for segment in segments:
            for match in pretokenize.finditer(segment):
                token_str = match.group()
                token_bytes = token_str.encode("utf-8")
                token_tuple = tuple(BYTES_CACHE[b] for b in token_bytes)
                
                counter[token_bytes] += 1

    q.put(counter)



def parallelize_pretokenization(
        input_path: str | os.PathLike,
        num_process: int,
        special_tokens: List[str],
)-> Counter[Tuple[bytes, ...]]:
    q = mp.Queue()
    processes = []
    total_count = Counter()

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_process, b"<|endoftext|>")

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            p = mp.Process(target= pretokenize_by_chunk, args=(input_path, start, end, special_tokens, q))
            processes.append(p)
        
        for p in processes:
            p.start()

        for _ in range(len(processes)):
            total_count += q.get()

        for p in processes:
            p.join()
        

    return total_count