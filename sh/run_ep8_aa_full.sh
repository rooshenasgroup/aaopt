for run in {1..1}
do
    seed=$((run + 40))
    echo "Outer loop iteration: $run (seed=$seed)"

    for sdiff_cons in 0.25
    do
        echo "sdiff_cons=$sdiff_cons"
        CUDA_VISIBLE_DEVICES="0" setsid torchrun --nproc_per_node=1 --master_port=27906 main_bpda_eot.py --dataset 'CIFAR10' --batch_size 8 \
            --network_pkl "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-vp.pkl" \
            --clf_net 'wideresnet-28-10' --subset_size 512 --data_seed $seed --seed $seed \
            --purify_model 'opt' --forward_steps 0.25 --total_steps 1 --purify_iter 20 \
            --att_method 'pgd_eot' --att_lp_norm -1 --att_eps 0.031373 --att_n_iter 20 --eot_attack_reps 20 \
            --lr 0.1 --purify_method 'x0' --sdiff_cons $sdiff_cons --ex_grad 1 --aa 1 --use_full_steps 1 --sdiff_net "pertubation_model_path"
        
        echo "Done with sdiff_cons=$sdiff_cons"
        echo " "
    done

    echo "Completed outer loop iteration: $run (seed=$seed)"
    echo "==============================="
done