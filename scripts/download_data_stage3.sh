data_path='/home/haibo/data'

# download VideoChat_instruct
mkdir ${data_path}/VideoChat_instruct
cd ${data_path}/VideoChat_instruct
for i in $(seq 1); do
    wget -O chunk_${i}.zip https://huggingface.co/datasets/WHB139426/Grounded-VideoLLM/resolve/main/VideoChat_instruct/chunk_${i}.zip
    unzip -o chunk_${i}.zip
    rm -rf chunk_${i}.zip
done

# download TextVR
mkdir ${data_path}/TextVR
cd ${data_path}/TextVR
for i in $(seq 1); do
    wget -O chunk_${i}.zip https://huggingface.co/datasets/WHB139426/Grounded-VideoLLM/resolve/main/TextVR/chunk_${i}.zip
    unzip -o chunk_${i}.zip
    rm -rf chunk_${i}.zip
done

# download Charades
mkdir ${data_path}/Charades
cd ${data_path}/Charades
for i in $(seq 1 2); do
    wget -O chunk_${i}.zip https://huggingface.co/datasets/WHB139426/Grounded-VideoLLM/resolve/main/Charades/chunk_${i}.zip
    unzip -o chunk_${i}.zip
    rm -rf chunk_${i}.zip
done

# download DiDeMo
mkdir ${data_path}/DiDeMo
cd ${data_path}/DiDeMo
for i in $(seq 1 2); do
    wget -O chunk_${i}.zip https://huggingface.co/datasets/WHB139426/Grounded-VideoLLM/resolve/main/DiDeMo/chunk_${i}.zip
    unzip -o chunk_${i}.zip
    rm -rf chunk_${i}.zip
done

# download HiREST
mkdir ${data_path}/HiREST
cd ${data_path}/HiREST
for i in $(seq 1 2); do
    wget -O chunk_${i}.zip https://huggingface.co/datasets/WHB139426/Grounded-VideoLLM/resolve/main/HiREST/chunk_${i}.zip
    unzip -o chunk_${i}.zip
    rm -rf chunk_${i}.zip
done

# download VTG-IT
mkdir ${data_path}/VTG-IT
cd ${data_path}/VTG-IT
for i in $(seq 1 9); do
    wget -O chunk_${i}.zip https://huggingface.co/datasets/WHB139426/Grounded-VideoLLM/resolve/main/VTG-IT/chunk_${i}.zip
    unzip -o chunk_${i}.zip
    rm -rf chunk_${i}.zip
done

# download activitynet
mkdir ${data_path}/activitynet
cd ${data_path}/activitynet
for i in $(seq 1 4); do
    wget -O chunk_${i}.zip https://huggingface.co/datasets/WHB139426/Grounded-VideoLLM/resolve/main/activitynet/chunk_${i}.zip
    unzip -o chunk_${i}.zip
    rm -rf chunk_${i}.zip
done

# download clevrer
mkdir ${data_path}/clevrer
cd ${data_path}/clevrer
for i in $(seq 1); do
    wget -O chunk_${i}.zip https://huggingface.co/datasets/WHB139426/Grounded-VideoLLM/resolve/main/clevrer/chunk_${i}.zip
    unzip -o chunk_${i}.zip
    rm -rf chunk_${i}.zip
done

# download coin
mkdir ${data_path}/coin
cd ${data_path}/coin
for i in $(seq 1 3); do
    wget -O chunk_${i}.zip https://huggingface.co/datasets/WHB139426/Grounded-VideoLLM/resolve/main/coin/chunk_${i}.zip
    unzip -o chunk_${i}.zip
    rm -rf chunk_${i}.zip
done

# download kinetics
mkdir ${data_path}/kinetics
cd ${data_path}/kinetics
for i in $(seq 1 2); do
    wget -O chunk_${i}.zip https://huggingface.co/datasets/WHB139426/Grounded-VideoLLM/resolve/main/kinetics/chunk_${i}.zip
    unzip -o chunk_${i}.zip
    rm -rf chunk_${i}.zip
done

# download msrvttqa
mkdir ${data_path}/msrvttqa
cd ${data_path}/msrvttqa
for i in $(seq 1); do
    wget -O chunk_${i}.zip https://huggingface.co/datasets/WHB139426/Grounded-VideoLLM/resolve/main/msrvttqa/chunk_${i}.zip
    unzip -o chunk_${i}.zip
    rm -rf chunk_${i}.zip
done

# download msvdqa
mkdir ${data_path}/msvdqa
cd ${data_path}/msvdqa
for i in $(seq 1); do
    wget -O chunk_${i}.zip https://huggingface.co/datasets/WHB139426/Grounded-VideoLLM/resolve/main/msvdqa/chunk_${i}.zip
    unzip -o chunk_${i}.zip
    rm -rf chunk_${i}.zip
done

# download nextqa
mkdir ${data_path}/nextqa
cd ${data_path}/nextqa
for i in $(seq 1); do
    wget -O chunk_${i}.zip https://huggingface.co/datasets/WHB139426/Grounded-VideoLLM/resolve/main/nextqa/chunk_${i}.zip
    unzip -o chunk_${i}.zip
    rm -rf chunk_${i}.zip
done

# download querYD
mkdir ${data_path}/querYD
cd ${data_path}/querYD
for i in $(seq 1 2); do
    wget -O chunk_${i}.zip https://huggingface.co/datasets/WHB139426/Grounded-VideoLLM/resolve/main/querYD/chunk_${i}.zip
    unzip -o chunk_${i}.zip
    rm -rf chunk_${i}.zip
done

# download sthsthv2
mkdir ${data_path}/sthsthv2
cd ${data_path}/sthsthv2
for i in $(seq 1); do
    wget -O chunk_${i}.zip https://huggingface.co/datasets/WHB139426/Grounded-VideoLLM/resolve/main/sthsthv2/chunk_${i}.zip
    unzip -o chunk_${i}.zip
    rm -rf chunk_${i}.zip
done

# download videochat2_conversations
mkdir ${data_path}/videochat2_conversations
cd ${data_path}/videochat2_conversations
for i in $(seq 1); do
    wget -O chunk_${i}.zip https://huggingface.co/datasets/WHB139426/Grounded-VideoLLM/resolve/main/videochat2_conversations/chunk_${i}.zip
    unzip -o chunk_${i}.zip
    rm -rf chunk_${i}.zip
done

# download videochat2_egoqa
mkdir ${data_path}/videochat2_egoqa
cd ${data_path}/videochat2_egoqa
for i in $(seq 1); do
    wget -O chunk_${i}.zip https://huggingface.co/datasets/WHB139426/Grounded-VideoLLM/resolve/main/videochat2_egoqa/chunk_${i}.zip
    unzip -o chunk_${i}.zip
    rm -rf chunk_${i}.zip
done

# download vitt
mkdir ${data_path}/vitt
cd ${data_path}/vitt
for i in $(seq 1 4); do
    wget -O chunk_${i}.zip https://huggingface.co/datasets/WHB139426/Grounded-VideoLLM/resolve/main/vitt/chunk_${i}.zip
    unzip -o chunk_${i}.zip
    rm -rf chunk_${i}.zip
done

# download youcook2
mkdir ${data_path}/youcook2
cd ${data_path}/youcook2
for i in $(seq 1 7); do
    wget -O chunk_${i}.zip https://huggingface.co/datasets/WHB139426/Grounded-VideoLLM/resolve/main/youcook2/chunk_${i}.zip
    unzip -o chunk_${i}.zip
    rm -rf chunk_${i}.zip
done

# download webvid-qa
mkdir ${data_path}/webvid-qa
cd ${data_path}/webvid-qa
for i in $(seq 1 7); do
    wget -O chunk_${i}.zip https://huggingface.co/datasets/WHB139426/Grounded-VideoLLM/resolve/main/webvid-qa/chunk_${i}.zip
    unzip -o chunk_${i}.zip
    rm -rf chunk_${i}.zip
done

# download webvid-caption
mkdir ${data_path}/webvid-caption
cd ${data_path}/webvid-caption
for i in $(seq 1 26); do
    wget -O chunk_${i}.zip https://huggingface.co/datasets/WHB139426/Grounded-VideoLLM/resolve/main/webvid-caption/chunk_${i}.zip
    unzip -o chunk_${i}.zip
    rm -rf chunk_${i}.zip
done

# download sharegpt4video
mkdir ${data_path}/sharegpt4video
cd ${data_path}/sharegpt4video
for i in $(seq 1 3); do
    wget -O chunk_${i}.zip https://huggingface.co/datasets/WHB139426/sharegpt4video-360p/resolve/main/chunk_${i}.zip
    unzip -o chunk_${i}.zip
    rm -rf chunk_${i}.zip
done


# download qvhighlights
mkdir ${data_path}/qvhighlights
cd ${data_path}/qvhighlights
for i in $(seq 1 5); do
    wget -O chunk_${i}.zip https://huggingface.co/datasets/WHB139426/Grounded-VideoLLM/resolve/main/qvhighlights/chunk_${i}.zip
    unzip -o chunk_${i}.zip
    rm -rf chunk_${i}.zip
done

# download mix_sft
mkdir ${data_path}/mix_sft
cd ${data_path}/mix_sft
wget https://huggingface.co/datasets/WHB139426/Grounded-VideoLLM/resolve/main/mix_sft/mix_sft.json

