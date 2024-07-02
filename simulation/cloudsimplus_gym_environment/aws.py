from abc import ABC, abstractmethod
from py4j.java_gateway import JavaGateway
from py4j.java_collections import ListConverter


class EC2(ABC):
    def __init__(self) -> None:
        self.family: str
        self.cpu: int
        self.memory: int
        self.price: float


class EC2Small(EC2):
    def __init__(self) -> None:
        self.family = "small"
        self.cpu = 1
        self.memory = 512
        self.price = 0.05
    

class EC2Medium(EC2):
    def __init__(self) -> None:
        self.family = "medium"
        self.cpu = 2
        self.memory = 1024
        self.price = 0.10


class EC2Large(EC2):
    def __init__(self) -> None:
        self.family = "large"
        self.cpu = 4
        self.memory = 2048
        self.price = 0.20


class AWS:
    def __init__(self, gateway, simulation, num_hosts, host_pes, vm_family, num_vms) -> None:
        self.vm = vm_family
        self.num_vms = num_vms
        self.simulation = simulation

        self.gateway = gateway

        self.datacenter = self.create_datacenter(num_hosts, host_pes)
        self.vms_list = ListConverter().convert(self.create_vms(), gateway._gateway_client)
        DatacenterBrokerSimple = gateway.jvm.org.cloudsimplus.brokers.DatacenterBrokerSimple
        self.broker = DatacenterBrokerSimple(simulation)


    def create_host(self, host_pes):
        PeSimple = self.gateway.jvm.org.cloudsimplus.resources.PeSimple
        HostSimple = self.gateway.jvm.org.cloudsimplus.hosts.HostSimple

        pe_list = []
        default_mips = float(1000)
        #List of Host's CPUs (Processing Elements, PEs)
        for _ in range(host_pes):
            pe_list.append(PeSimple(default_mips))
        pe_list = ListConverter().convert(pe_list, self.gateway._gateway_client)
        ram = 4096 #in Megabytes
        bw = 10000 #in Megabits/s
        storage = 1000000 #in Megabytes

        # Uses ResourceProvisionerSimple by default for RAM and BW provisioning
        # and VmSchedulerSpaceShared for VM scheduling.
        host = HostSimple(ram, bw, storage, pe_list)

        return host


    def create_datacenter(self, num_hosts, host_pes):
        DatacenterSimple = self.gateway.jvm.org.cloudsimplus.datacenters.DatacenterSimple

        host_list = []
        for _ in range(num_hosts):
            host = self.create_host(host_pes);
            host_list.append(host)
        host_list = ListConverter().convert(host_list, self.gateway._gateway_client)
        #Uses a VmAllocationPolicySimple by default to allocate VMs
        return DatacenterSimple(self.simulation, host_list)

    def create_vms(self):
        VmSimple = self.gateway.jvm.org.cloudsimplus.vms.VmSimple
        list = []

        for _ in range(self.num_vms):
            #Uses a CloudletSchedulerTimeShared by default to schedule Cloudlets
            vm = VmSimple(float(1000), self.vm.cpu)
            vm.setRam(self.vm.memory).setBw(1000).setSize(10000)
            list.append(vm)
        return list
    

    def get_vm_memory_percent_usage(self):
        memory_list = []
        for vm in self.vms_list:
            memory_list.append(vm.getRam().getPercentUtilization())
        return memory_list
    

    def get_vm_cpu_percent_usage(self):
        cpu_list = []
        for vm in self.vms_list:
            cpu_list.append(vm.getCpuPercentUtilization())
        return cpu_list