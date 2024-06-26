sudo sh -c "echo never >/sys/kernel/mm/transparent_hugepage/enabled"
sudo sh -c "echo '-1' >/proc/sys/kernel/perf_event_paranoid"
sudo systemctl disable irqbalance
sudo systemctl stop irqbalance
sudo sh -c "echo 1 > /sys/devices/system/cpu/intel_pstate/no_turbo"
#sudo vi /etc/default/irqbalance # set banned cpu list to the atom cores.
