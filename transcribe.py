import time
import concurrent.futures
from utils import shrink_and_split_mp3
import io
import torch
from faster_whisper import WhisperModel
from tqdm import tqdm
from pydub import AudioSegment
from utils import chunk_text_into_sentences

def infer_from_replicate(replicate_client, mp3_file):
    # Read the local audio file in binary mode
    with open(mp3_file, "rb") as f:
        audio_blob = io.BytesIO(f.read())  # Use BytesIO to create a file-like object

    # create sepaarte function
    output = replicate_client.run(
        "vaibhavs10/incredibly-fast-whisper:3ab86df6c8f54c11309d4d1f930ac292bad43ace52d10c80d87eb258b3c9f79c",
        input={
            "task": "transcribe",
            "audio": audio_blob,
            "language": "None",
            "timestamp": "chunk",
            "batch_size": 64,
            "diarise_audio": False
        }
    )

    return output

def transcribe_with_replicate(replicate_client, mp3_file, n_splits=2):
    start_time = time.time()
    # shrink and split mp3 - return list of partial episodes
    mp3_files = shrink_and_split_mp3(mp3_file[0], n_splits)
    # Using ThreadPoolExecutor to parallelize downloads
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:        
        # Submit the transcription tasks to the executor
        futures = {
            executor.submit(infer_from_replicate, replicate_client, mp3_file): i for i, mp3_file in enumerate(mp3_files)
        }
        # Collect the results in the original order based on submission index
        results = [None] * len(mp3_files)
        for future in concurrent.futures.as_completed(futures):
            i = futures[future]  # Index of the mp3 file part
            results[i] = future.result()  # Store result in the correct index
    chunks = []
    text = ''
    for output in results:
        chunks += output['chunks']
        text = " ".join([text, output['text']])
    for i, chunk in enumerate(chunks):
        chunk['id'] = str(i+1)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Time to run the command: {execution_time} seconds")
    return chunks, text

def infer_from_whistler(file_path):
  # define our torch configuration
  device = "cuda" if torch.cuda.is_available() else "cpu"
  # compute_type = "float16" if torch.cuda.is_available() else "float32"
  compute_type = "int8"

  # load model on GPU if available, else cpu
  model = WhisperModel("distil-large-v3", device=device, compute_type=compute_type)
    
  # fast whisper large 3
  final_transcription = ""
  segments, info = model.transcribe(file_path, beam_size=1)

  # Initialize the progress bar
  pbar = tqdm(total=len(AudioSegment.from_file(file_path)) / 1000.0, unit='s')

  for segment in segments:
      final_transcription += segment.text
      pbar.update(segment.end - segment.start)

  # Close the progress bar
  pbar.close()

  # Chunk the transcription text into sentences
  sentences = chunk_text_into_sentences(final_transcription)

  chunks = []
  for sentence in sentences:
      tmp_dict = {}
      tmp_dict['text'] = sentence
      chunks.append(tmp_dict)
        
  print("Audio transcription complete")
  data = {"chunks": chunks, "text": final_transcription}
  return data

def transcribe_with_whistler(filenames, n_splits=2):
    start_time = time.time()
    mp3_files = shrink_and_split_mp3(filenames[0], n_splits)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_splits) as executor:        
        # Submit the transcription tasks to the executor
        futures = {
            executor.submit(infer_from_whistler, mp3_file): i for i, mp3_file in enumerate(mp3_files)
        }
        # Collect the results in the original order based on submission index
        results = [None] * len(mp3_files)
        for future in concurrent.futures.as_completed(futures):
            i = futures[future]  # Index of the mp3 file part
            results[i] = future.result()  # Store result in the correct index
    chunks = []
    text = ''
    for output in results:
        chunks += output['chunks']
        text = " ".join([text, output['text']])
    for i, chunk in enumerate(chunks):
        chunk['id'] = str(i+1)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Time to run the command: {execution_time} seconds")
    return chunks, text