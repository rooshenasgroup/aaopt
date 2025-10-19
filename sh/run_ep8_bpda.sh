for run in {1..5}
do
    seed=$((run + 40))
    echo "Outer loop iteration: $run (seed=$seed)"

    for sdiff_cons in 0.5
    do
        echo "sdiff_cons=$sdiff_cons"
        CUDA_VISIBLE_DEVICES="3" setsid torchrun --nproc_per_node=1 --master_port=27806 main_bpda_eot.py --dataset 'CIFAR10' --batch_size 8 \
            --network_pkl "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-vp.pkl" \
            --clf_net 'wideresnet-28-10' --subset_size 512 --data_seed $seed --seed $seed \
            --purify_model 'opt' --forward_steps 0.5 --total_steps 1 --purify_iter 5 \
            --att_method 'bpda_eot' --att_lp_norm -1 --att_eps 0.031373 --att_n_iter 50 --eot_attack_reps 15 \
            --lr 0.1 --purify_method 'aaopt' --sdiff_cons $sdiff_cons --seq_t 1 --sdiff_net "./pretrained_models/cifar10_score_DIFF_JOINT_wideresnet-28-10_"
        
        echo "Done with sdiff_cons=$sdiff_cons"
        echo " "
    done

    echo "Completed outer loop iteration: $run (seed=$seed)"
    echo "==============================="
done