for run in {1..1}
do
    seed=$((run + 40))
    echo "Outer loop iteration: $run (seed=$seed)"

    for sdiff_cons in 0.2
    do
        bs=32

        echo "sdiff_cons=$sdiff_cons"

        CUDA_VISIBLE_DEVICES="0" setsid torchrun --nproc_per_node=1 --master_port=29933 main_transfer.py --dataset 'CIFAR10' --batch_size $bs \
            --network_pkl "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-vp.pkl" \
            --clf_net 'clip' --subset_size 512 --data_seed $seed --seed $seed \
            --purify_model 'opt' --forward_steps 0.1 --total_steps 1 --purify_iter 20 \
            --att_method 'clf_pgd' --att_lp_norm -1 --att_eps 0.031373 --att_step 40 \
            --lr 0.01 --purify_method 'x0' --sdiff_cons $sdiff_cons --seq_t 1 --sdiff_net "pertubation_model_path"
        
        echo "Done with sdiff_cons=$sdiff_cons"
        echo " "  # Print an empty new line for readability
    done

    echo "Completed outer loop iteration: $run (seed=$seed)"
    echo "==============================="
done