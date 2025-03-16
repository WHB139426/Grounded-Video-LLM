data_path='/home/haibo/data'

# download InternVid-G
mkdir ${data_path}/InternVid-G
cd ${data_path}/InternVid-G
for i in $(seq 1 7); do
    wget -O chunk_${i}.zip https://huggingface.co/datasets/WHB139426/Grounded-VideoLLM/resolve/main/InternVid-G/chunk_${i}.zip
    unzip -o chunk_${i}.zip
    rm -rf chunk_${i}.zip
done

# download vtimellm_stage2
mkdir ${data_path}/vtimellm_stage2
cd ${data_path}/vtimellm_stage2
for i in $(seq 1 10); do
    wget -O chunk_${i}.zip https://huggingface.co/datasets/WHB139426/Grounded-VideoLLM/resolve/main/vtimellm_stage2/chunk_${i}.zip
    unzip -o chunk_${i}.zip
    rm -rf chunk_${i}.zip
done

# download Moment-10m
mkdir ${data_path}/Moment-10m
cd ${data_path}/Moment-10m
for i in $(seq 1 49); do
    wget -O chunk_${i}.zip https://huggingface.co/datasets/WHB139426/Grounded-VideoLLM/resolve/main/Moment-10m/chunk_${i}.zip
    unzip -o chunk_${i}.zip
    rm -rf chunk_${i}.zip
done

# download mix_grounded
mkdir ${data_path}/mix_grounded
cd ${data_path}/mix_grounded
wget https://huggingface.co/datasets/WHB139426/Grounded-VideoLLM/resolve/main/mix_grounded/mix_grounded.json