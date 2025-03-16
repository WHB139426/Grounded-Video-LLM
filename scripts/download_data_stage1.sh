data_path='/home/haibo/data'

# download webvid-703k
mkdir ${data_path}/webvid-703k
cd ${data_path}/webvid-703k
# 循环下载 chunk_1.zip 到 chunk_15.zip
for i in $(seq 1 15); do
    wget -O chunk_${i}.zip https://huggingface.co/datasets/WHB139426/Grounded-VideoLLM/resolve/main/webvid-703k/chunk_${i}.zip
    unzip -o chunk_${i}.zip
    rm -rf chunk_${i}.zip
done

# download internvid
mkdir ${data_path}/internvid
cd ${data_path}/internvid
for i in $(seq 1 10); do
    wget -O chunk_${i}.zip https://huggingface.co/datasets/WHB139426/Grounded-VideoLLM/resolve/main/internvid/chunk_${i}.zip
    unzip -o chunk_${i}.zip
    rm -rf chunk_${i}.zip
done

# download panda70m_2m
mkdir ${data_path}/panda70m_2m
cd ${data_path}/panda70m_2m
for i in $(seq 1 25); do
    wget -O chunk_${i}.zip https://huggingface.co/datasets/WHB139426/Grounded-VideoLLM/resolve/main/panda70m_2m/chunk_${i}.zip
    unzip -o chunk_${i}.zip
    rm -rf chunk_${i}.zip
done

# download mix_pretrain
mkdir ${data_path}/mix_pretrain
cd ${data_path}/mix_pretrain
wget https://huggingface.co/datasets/WHB139426/Grounded-VideoLLM/resolve/main/mix_pretrain/mix_pretrain.json