import os

from sentence_transformers import SentenceTransformer
import time
import torch
import csv


if __name__ == '__main__':

    print(torch.__version__)

    if torch.cuda.is_available():
        device = "cuda"  # buffer must be large enough
    elif torch.backends.mps.is_available():
        device = "mps"  # buffer must be large enough
    else:
        device = "cpu"  # buffer must be small

    print(device)

    bufferSize = 3 * 4000  # 3 - name, description, labels

    start_time = time.time()
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Load model: {elapsed_time} seconds")

    fileEmbeddings = 'embeddings.csv'
    fileLastSavedItemId = 'lastSavedItemId.txt'

    current_dir = os.path.dirname(os.path.realpath(__file__)) + '/'

    # if file not exists, create it and add 0
    if not os.path.exists(current_dir + fileLastSavedItemId):
        with open(fileLastSavedItemId, 'w') as f:
            f.write('0')


    # load last saved item id
    with open(current_dir + fileLastSavedItemId, 'r') as f:
        lastSavedItemIdStr = f.read()
        lastSavedItemId = int(lastSavedItemIdStr)

    fileCsv = 'dataset.csv'
    preprocessed = 0
    timeConsumed = 0
    with open(current_dir + fileCsv, 'r', newline='') as csvfile:
        csvreader = csv.reader(csvfile)

        # Skip the header
        next(csvreader)

        buffer = []
        ids = []
        errors = 0
        # Iterate over each row after the header
        for row in csvreader:
            try:
                ID, name, description, labels = row
                ID = int(ID)
            except ValueError:
                print("Error in row: ", row)
                errors += 1
                continue

            if ID <= lastSavedItemId:
                continue

            ids.append(ID)
            buffer.append(name)
            buffer.append(description)
            buffer.append(labels)

            if len(buffer) >= bufferSize:
                timeStart = time.time()
                results = model.encode(
                    buffer,
                    normalize_embeddings=True, batch_size=64, device=device
                )

                # split results by 3 items
                for i in range(0, len(results), 3):
                    name_embedding = results[i]
                    description_embedding = results[i + 1]
                    labels_embedding = results[i + 2]

                    with open(current_dir + fileEmbeddings, 'a', newline='') as f:
                        itemId = ids[i // 3]
                        writer = csv.writer(f)
                        writer.writerow(
                            [itemId, name_embedding.tolist(), description_embedding.tolist(),
                             labels_embedding.tolist()])
                        with open(current_dir + fileLastSavedItemId, 'w') as fs:
                            fs.write(str(itemId))

                preprocessed += bufferSize / 3

                timeEnd = time.time()
                timeConsumed += timeEnd - timeStart
                print(f"Total processed: {preprocessed}")
                print(f"Time consumed: {timeConsumed} seconds")
                print(f"Errors: {errors}")
                buffer = []
                ids = []

        if len(buffer) > 0:
            results = model.encode(
                buffer,
                normalize_embeddings=True, batch_size=64, device=device
            )

            # split results by 3 items
            for i in range(0, len(results), 3):
                name_embedding = results[i]
                description_embedding = results[i + 1]
                labels_embedding = results[i + 2]

                with open(current_dir + fileEmbeddings, 'a', newline='') as f:
                    itemId = ids[i // 3]
                    writer = csv.writer(f)
                    writer.writerow(
                        [itemId, name_embedding.tolist(), description_embedding.tolist(),
                         labels_embedding.tolist()])
                    with open(current_dir + fileLastSavedItemId, 'w') as fs:
                        fs.write(str(itemId))

            preprocessed += len(buffer) / 3


    print(f"Total processed: {preprocessed}")

    # gzip the file
    print("Gzipping the file")
    os.system(f'gzip -f {fileEmbeddings}')
    print("Gzipping done")
