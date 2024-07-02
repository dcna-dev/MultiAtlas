from py4j.java_gateway import JavaGateway
from py4j.java_collections import ListConverter


from aws import AWS, EC2Small, EC2Medium, EC2Large


gateway = JavaGateway()

CloudletSimple          = gateway.jvm.org.cloudsimplus.cloudlets.CloudletSimple
UtilizationModelFull    = gateway.jvm.org.cloudsimplus.utilizationmodels.UtilizationModelFull


def create_cloudlets(cloudlets, cloudlet_length, cloudlet_pes):
    list = []

    for i in range(cloudlets):
        cloudlet = CloudletSimple(cloudlet_length, cloudlet_pes)
        cloudlet.setUtilizationModelCpu(UtilizationModelFull()).setSizes(1024).setSubmissionDelay(float(i))
        list.append(cloudlet)

    return list;


simulation = gateway.jvm.org.cloudsimplus.core.CloudSimPlus()
vm_family = EC2Large()

aws_provider = AWS(gateway, simulation, 2, 4, vm_family, 4)

INTERVAL = 600
CLOUDLETS = 8
CLOUDLET_PES = 2
CLOUDLET_LENGTH = 10000
cloudlet_list = create_cloudlets(CLOUDLETS, CLOUDLET_LENGTH, CLOUDLET_PES)
cloudlet_list = ListConverter().convert(cloudlet_list, gateway._gateway_client)

aws_provider.broker.submitVmList(aws_provider.vms_list)
aws_provider.broker.submitCloudletList(cloudlet_list)

simulation.startSync()

while simulation.isRunning():
    simulation.runFor(float(INTERVAL))
    print(f"CPU usage: {aws_provider.get_vm_cpu_percent_usage()}")
    print(f"RAM usage: {aws_provider.get_vm_memory_percent_usage()}")

