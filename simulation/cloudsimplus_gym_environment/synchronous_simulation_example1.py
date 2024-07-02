from py4j.java_gateway import JavaGateway
from py4j.java_collections import ListConverter



gateway = JavaGateway()

# Adding referencies CloudSimPlus java dependencies

HostFactory             = gateway.jvm.org.cloudsimplus.util.HostFactory
DatacenterBroker        = gateway.jvm.org.cloudsimplus.brokers.DatacenterBroker
DatacenterBrokerSimple  = gateway.jvm.org.cloudsimplus.brokers.DatacenterBrokerSimple
CloudletsTableBuilder   = gateway.jvm.org.cloudsimplus.builders.tables.CloudletsTableBuilder
Cloudlet                = gateway.jvm.org.cloudsimplus.cloudlets.Cloudlet
CloudletSimple          = gateway.jvm.org.cloudsimplus.cloudlets.CloudletSimple
CloudSimPlus            = gateway.jvm.org.cloudsimplus.core.CloudSimPlus
Datacenter              = gateway.jvm.org.cloudsimplus.datacenters.Datacenter
DatacenterSimple        = gateway.jvm.org.cloudsimplus.datacenters.DatacenterSimple
Host                    = gateway.jvm.org.cloudsimplus.hosts.Host
HostSimple              = gateway.jvm.org.cloudsimplus.hosts.HostSimple
Pe                      = gateway.jvm.org.cloudsimplus.resources.Pe
PeSimple                = gateway.jvm.org.cloudsimplus.resources.PeSimple
Log                     = gateway.jvm.org.cloudsimplus.util.Log
UtilizationModelFull    = gateway.jvm.org.cloudsimplus.utilizationmodels.UtilizationModelFull
Vm                      = gateway.jvm.org.cloudsimplus.vms.Vm
VmSimple                = gateway.jvm.org.cloudsimplus.vms.VmSimple

Comparator              = gateway.jvm.java.util.Comparator
Math                    = gateway.jvm.java.lang.Math



def create_host(host_pes):
    pe_list = []
    default_mips = float(1000)
    #List of Host's CPUs (Processing Elements, PEs)
    for _ in range(host_pes):
        pe_list.append(PeSimple(default_mips))
    pe_list = ListConverter().convert(pe_list, gateway._gateway_client)
    ram = 2048 #in Megabytes
    bw = 10000 #in Megabits/s
    storage = 1000000 #in Megabytes

    # Uses ResourceProvisionerSimple by default for RAM and BW provisioning
    # and VmSchedulerSpaceShared for VM scheduling.
    host = HostSimple(ram, bw, storage, pe_list)

    return host


def create_datacenter(hosts, host_pes):
    host_list = []
    for _ in range(hosts):
        host = create_host(host_pes);
        host_list.append(host)
    host_list = ListConverter().convert(host_list, gateway._gateway_client)
    #Uses a VmAllocationPolicySimple by default to allocate VMs
    return DatacenterSimple(simulation, host_list)


def create_vms(vms, vm_pes):
    list = []

    for _ in range(vms):
        #Uses a CloudletSchedulerTimeShared by default to schedule Cloudlets
        vm = VmSimple(float(1000), vm_pes)
        vm.setRam(512).setBw(1000).setSize(10000)
        list.append(vm)

    return list


def create_cloudlets(cloudlets, cloudlet_length, cloudlet_pes):
    list = []

    for i in range(cloudlets):
        cloudlet = CloudletSimple(cloudlet_length, cloudlet_pes)
        cloudlet.setUtilizationModelCpu(UtilizationModelFull()).setSizes(1024).setSubmissionDelay(float(i))
        list.append(cloudlet)

    return list;


def print_vm_cpu_utilization(simulation, vm_list):
    #import pdb; pdb.set_trace()
    global PREVIOUS_CLOCK
    # To avoid printing to much data, just prints if the simulation clock
    # has changed, it's multiple of the interval to increase clock
    # and there is some VM already running.
    #if simulation.clock() == PREVIOUS_CLOCK or Math.round(simulation.clock()) % INTERVAL != 0 or broker0.getVmExecList().isEmpty():
    #    return

    PREVIOUS_CLOCK = simulation.clock()
    print(f"VM CPU utilization for Time {simulation.clock()}")
    for vm in vm_list:
        print(f" Vm {vm.getId()}", end="|")
    print()

    for vm in vm_list:
        print(f" {vm.getCpuPercentUtilization()*100}", end="|")
    print("")


INTERVAL = 600
HOSTS = 2
HOST_PES = 4
VMS = 4
VM_PES = 2
CLOUDLETS = 8
CLOUDLET_PES = 2
CLOUDLET_LENGTH = 10000
PREVIOUS_CLOCK = -1

simulation = CloudSimPlus()

datacenter0 = create_datacenter(HOSTS, HOST_PES)

#Creates a broker that is a software acting on behalf of a cloud customer to manage his/her VMs and Cloudlets
broker0 = DatacenterBrokerSimple(simulation)

vm_list = create_vms(VMS, VM_PES)
vm_list = ListConverter().convert(vm_list, gateway._gateway_client)
cloudlet_list = create_cloudlets(CLOUDLETS, CLOUDLET_LENGTH, CLOUDLET_PES)
cloudlet_list = ListConverter().convert(cloudlet_list, gateway._gateway_client)
broker0.submitVmList(vm_list)
broker0.submitCloudletList(cloudlet_list)

simulation.startSync()

while simulation.isRunning():
    simulation.runFor(float(INTERVAL))
    print_vm_cpu_utilization(simulation, vm_list)

cloudletFinishedList = broker0.getCloudletFinishedList()
#cloudletFinishedList.sort(Comparator.comparingLong(Cloudlet::getId))
CloudletsTableBuilder(cloudletFinishedList).build()