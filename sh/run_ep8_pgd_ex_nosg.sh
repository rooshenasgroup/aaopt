for run in {1..1}
do
    seed=$((run + 40))
    echo "Outer loop iteration: $run (seed=$seed)"

    for sdiff_cons in 0.25
    do
        bs=4
        eot=20

        echo "sdiff_cons=$sdiff_cons"

        CUDA_VISIBLE_DEVICES="3" setsid torchrun --nproc_per_node=1 --master_port=29133 main_bpda_eot.py --dataset 'CIFAR10' --batch_size $bs \
            --network_pkl "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-vp.pkl" \
            --clf_net 'wideresnet-28-10' --subset_size 512 --data_seed $seed --seed $seed \
            --purify_model 'opt' --forward_steps 0.25 --total_steps 1 --purify_iter 20 --apprx_iter 1 --ex_grad 1 --no_stop_grad 1 \
            --att_method 'pgd_eot' --att_lp_norm -1 --att_eps 0.031373 --att_n_iter 20 --eot_attack_reps $eot \
            --lr 0.1 --purify_method 'aaopt' --sdiff_cons $sdiff_cons --sdiff_net "./pretrained_models/cifar10_score_DIFF_JOINT_wideresnet-28-10_"
        
        echo "Done with sdiff_cons=$sdiff_cons"
        echo " "  # Print an empty new line for readability
    done

    echo "Completed outer loop iteration: $run (seed=$seed)"
    echo "==============================="
done