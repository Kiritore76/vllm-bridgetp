#!/usr/bin/env bash
# BridgeTP 服务器当前任务（唯一入口，每次下发新任务时覆盖本文件）。
# 运行：
#   cd /root/autodl-tmp/bridgetp/vllm_bridge && git pull --ff-only && bash tools/bridge_tp/server_run.sh
#
# 当前任务：数据盘已满（qwen3_fast_conversion_20261101_230739_24226 的 tilt-record 写盘失败）。
# 本脚本只做磁盘维护，不启动GPU实验。
#   bash tools/bridge_tp/server_run.sh        只报告：磁盘占用、最大的结果目录、可清理的目录；不删除。
#   bash tools/bridge_tp/server_run.sh clean  删除以下两类，然后打包 prestart-roll：
#     1) 已经拿回本地并核验过哈希的旧批次的原始目录 results/<批次>/
#        （只在同名外层包 results/<批次>.tar.gz 仍在服务器上时才删；外层包本身保留）；
#     2) 本批写盘失败、不完整的 tilt-record 原始目录（之后重跑）。
# 空间够（≥1GB）时把本批 prestart-roll 打包成外层包（不含两个场景的原始目录）。

results=/root/autodl-tmp/bridgetp/results
run=$results/qwen3_fast_conversion_20261101_230739_24226
mode=${1:-report}
[ "$mode" = report ] || [ "$mode" = clean ] || { echo "参数只能是 report 或 clean"; exit 1; }

# 已拿回本地并核验过的批次（multi/results 里有对应外层包）。
retrieved=(
    qwen3_cleanup_verify_20261029_085117_16753
    qwen3_conversion_assist_20261101_093941_20421
    qwen3_conversion_assist_20261101_095943_13279
    qwen3_conversion_assist_20261101_103558_4576
    qwen3_conversion_assist_20261101_111729_3221
    qwen3_conversion_assist_20261101_115931_15409
    qwen3_conversion_assist_20261101_125259_23252
    qwen3_conversion_join_20261101_204155_31036
    qwen3_conversion_live_20261031_195116_9379
    qwen3_conversion_live_20261031_200740_31723
    qwen3_conversion_live_20261031_202458_18532
    qwen3_conversion_shadow_20261031_213909_7335
    qwen3_decider_record_20261101_215723_21941
    qwen3_deep_stay_20261030_235237_26005
    qwen3_diverse_long12_20261027_131520_29540
    qwen3_feature_probe2_20261027_124250_8420
    qwen3_formal16k_20261027_152412_29234
    qwen3_four_mode_20261029_141957_27701
    qwen3_four_mode_complete_20261029_161654_11365
    qwen3_g1_three_card_20261031_132802_17905
    qwen3_g2_l1a_channels_20261031_155228_31310
    qwen3_g2_l1a_channels_20261031_160539_31193
    qwen3_g2_l1a_channels_20261031_162721_17395
    qwen3_g2_l1a_channels_20261031_163818_7658
    qwen3_guard384_20261029_201207_10985
    qwen3_history_memory_20261030_231324_22964
    qwen3_import_timing_20261031_110842_24286
    qwen3_joint_kv_20261030_112055_31617
    qwen3_kv_route_20261030_134013_13201
    qwen3_link_empty_drain_20261031_104100_28261
    qwen3_loaded_drain_20261031_171448_29158
    qwen3_long208_four_gpu_20261028_120902_19080
    qwen3_long_burst_20261030_220059_32696
    qwen3_migration128_retry_20261028_184040_15099
    qwen3_migration_matrix_20261029_171524_1079
    qwen3_online_loaded_20261028_213640_29491
    qwen3_parameter_batch_20261028_161242_17847
    qwen3_pressure_four_20261029_224513_30152
    qwen3_pressure_observe_20261029_111433_28547
    qwen3_rate_sweep_20261101_165702_28846
    qwen3_rate_sweep_hi_20261101_183153_12046
    qwen3_ready_slowdown_20261028_224103_17911
    qwen3_recipe12_v1_20261027_124939_31646
    qwen3_resident_pressure_20261029_211618_8861
    qwen3_retrain8k62_a100style_20261028_112539_27464
    qwen3_retrain8k62_merged2208_20261028_135855_692
    qwen3_routine_four_20261030_190853_23460
    qwen3_runtime_fix_20261029_000714_25481
    qwen3_scheduler_supplement_20261028_185441_12295
    qwen3_static_layout_screen_20261101_133349_1180
    qwen3_static_layout_screen_20261101_154945_23093
    qwen3_stay_migrate_20261029_123432_23373
    qwen3_tp2_service_20261028_151240_21544
    qwen3_train8k62_20261028_073549_28433
    qwen3_updated_four_20261030_142940_15831
    qwen3_urgent_admission_20261030_123530_11697
)

echo '=== 数据盘 ==='
df -h /root/autodl-tmp
echo '=== results 下最大的25项 ==='
du -sh "$results"/* 2>/dev/null | sort -h | tail -25

candidates=()
for name in "${retrieved[@]}"; do
    if [ -d "$results/$name" ] && { [ -f "$results/$name.tar.gz" ] || [ -f "$results/$name.compact.tar.gz" ]; }; then
        candidates+=("$results/$name")
    fi
done
[ -d "$run/tilt-record" ] && candidates+=("$run/tilt-record")
echo '=== 可清理的目录（clean模式才删除） ==='
if [ ${#candidates[@]} = 0 ]; then
    echo '（没有）'
else
    du -sh "${candidates[@]}" 2>/dev/null
    du -csh "${candidates[@]}" 2>/dev/null | tail -1
fi

if [ "$mode" = clean ]; then
    for path in "${candidates[@]}"; do
        echo "删除：$path"
        rm -rf -- "$path"
    done
    rm -f -- "$run/tilt-record.console.log" "$run/tilt-record.started_unix_s"
    echo '=== 清理后 ==='
    df -h /root/autodl-tmp
fi

avail=$(df --output=avail -k /root/autodl-tmp | tail -1)
if [ "$avail" -lt 1048576 ]; then
    echo '可用空间不足1GB，未打包；看过上面的列表后可运行：bash tools/bridge_tp/server_run.sh clean'
    exit 2
fi
[ -f "$run/prestart-roll.tar.gz" ] || { echo "缺少 $run/prestart-roll.tar.gz，未打包"; exit 1; }
echo '=== 打包 prestart-roll ==='
tar -C "$results" -czf "$run.tar.gz" \
    --exclude="$(basename "$run")/prestart-roll" \
    --exclude="$(basename "$run")/tilt-record" \
    "$(basename "$run")" || exit 1
sha256sum "$run.tar.gz" > "$run.tar.gz.sha256"
ls -la "$run.tar.gz"
cat "$run.tar.gz.sha256"
echo "请拿回：$run.tar.gz"
echo "以及：$run.tar.gz.sha256"
echo 'tilt-record 需要在空间足够后重跑（下一次任务会单独下发）。'
