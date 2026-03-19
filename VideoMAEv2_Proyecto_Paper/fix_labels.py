import os
import glob

def fix_labels(split):
    old_csv = f'dataset/k400_luis/labels/{split}.csv'
    new_csv = f'dataset/k400_luis/labels/{split}_fixed.csv'
    cluster_dir = f'/home/datasets/k400/{split}'
    
    print(f'Loading available files in {cluster_dir}...')
    files_in_dir = os.listdir(cluster_dir)
    # create mapping from 11-char youtube ID to full filename
    id_to_file = {}
    for f in files_in_dir:
        if f.endswith('.mp4'):
            # Some files might not be 11 chars, but most K400 videos are <11_chars>_start_end.mp4
            # Actually, let's just take the string up to the first '_' or just first 11 chars.
            # But wait, youtube IDs can contain '_'. So the suffix is always _%06d_%06d.mp4.
            # Let's just use the first 11 characters as the YouTube ID.
            yt_id = f[:11]
            id_to_file[yt_id] = f

    print(f'Found {len(id_to_file)} unique videos in directory.')
    
    missing = 0
    found = 0
    with open(old_csv, 'r') as fin, open(new_csv, 'w') as fout:
        for line in fin:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            path_part, label = parts
            # path_part is like 'k400/ytID.mp4'  or 'ytID.mp4'
            filename = os.path.basename(path_part)
            yt_id = filename.replace('.mp4', '')
            # If the yt_id in the original csv has more than 11 chars, handle it? 
            # The original labels just used ytID.mp4
            if yt_id in id_to_file:
                actual_file = id_to_file[yt_id]
                # Write the new format: k400/actual_file.mp4 label
                fout.write(f'k400/{actual_file} {label}\n')
                found += 1
            else:
                # keep original or skip? Let's just keep original, it will error out if not skip, but if we drop it we lose acc?
                # actually, some might be in replacement? If not there, the dataset is just missing some videos.
                # It's better to omit videos that don't exist to avoid crashing the whole job.
                missing += 1

    print(f'[{split}] Fixed {found} files. Missing {missing} files. Saved to {new_csv}')

fix_labels('val')
# If train is needed we can do it too, but user is evaluating.
