import pykube
import base64
import logging
import pandas as pd
import os
import time

from datetime import datetime, date
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
from graceful_shutdown import ShutdownProtection


from datasource import Datasource, GrafanaQuery

class Monitor:
    def __init__(self, name, api_version, kind, influx_url, influx_token, influx_org, influx_bucket):
        self.name = name
        self.api_version = api_version
        self.kind = kind
        self.influx_url = influx_url
        self.influx_token = influx_token
        self.influx_org = influx_org
        self.influx_bucket = influx_bucket

        # Initialize the pykube
        kube_config = pykube.KubeConfig.from_env()
        self.kube_api = pykube.HTTPClient(kube_config)
        self.frequency = self.get_resource().obj["spec"]["frequency"]

        # Initialize the InfluxDB client
        self.influx_client = InfluxDBClient(url=self.influx_url, token=self.influx_token)
        self.influx_write_api = self.influx_client.write_api(write_options=SYNCHRONOUS)
        self.influx_query_api = self.influx_client.query_api()

    def get_resource(self):
        return pykube.object_factory(self.kube_api, self.api_version, self.kind).objects(self.kube_api).get_by_name(self.name)

    def get_autonomous_resources(self, api_version):
        return pykube.object_factory(self.kube_api, api_version, "AutonomousResource")
    
    def get_autonomous_resources_with_monitor_enabled(self, api_version):
        #import pdb; pdb.set_trace()
        autonomous_resources = []
        for resource in list(pykube.object_factory(self.kube_api, api_version, "AutonomousResource").objects(self.kube_api).iterator()):
            if bool(resource.obj["metadata"]["annotations"]["dcna.dev/monitor"]):
                autonomous_resources.append(resource)
        
        return autonomous_resources
    
    def decode_base64(self, encoded_string):
        base64_bytes = encoded_string.encode('ascii')
        message_bytes = base64.b64decode(base64_bytes)
        decoded_string = message_bytes.decode('ascii')
        return decoded_string

    def get_autonomous_resource_config(self, autonomous_resource):
        return autonomous_resource.obj["spec"]
    
    def collect_metrics_from_ds(self, autonomous_resource):
        monitor_config = self.get_autonomous_resource_config(autonomous_resource)["datasource"]
        datasource = pykube.object_factory(self.kube_api, "dcna.dev/v1beta1", 
                                           "GrafanaDatasource").objects(self.kube_api).get_by_name(monitor_config["name"])
        datasource_config = datasource.obj["spec"]
        datasource_secret = pykube.object_factory(self.kube_api, "v1", 
                                            "Secret").objects(self.kube_api).get_by_name(datasource_config["tokenSecretRef"]["name"])
        token = self.decode_base64(datasource_secret.obj["data"][datasource_config["tokenSecretRef"]["key"]])
        gds = Datasource(datasource_config["name"], datasource_config["base_url"], token)

        start_date = "now"
        end_date = f"now-{self.frequency}"
        metrics_name = []
        queries = []

        for metric in monitor_config["metrics"]:
            metrics_name.append(metric["name"])
            query_filters = ["metric.type", "=", metric["metric_type"]]
            grafana_query = GrafanaQuery(gds.info['type'], gds.info['uid'], metric["name"])
            queries.append(grafana_query.create_query(gds.info['jsonData']['defaultProject'], query_filters))

        result = gds.get_data(start_date, end_date, queries)
        
        return result
    
    def save_metrics(self, result, autonomous_resource):
        monitor_config = self.get_autonomous_resource_config(autonomous_resource)["datasource"]
        
        for metric in result['results'].keys():
            for i in range(len(monitor_config["metrics"])):
                if monitor_config["metrics"][i]['name'] == metric:
                    measurement = monitor_config["metrics"][i]['name']
            autonomous_resource_name = self.get_autonomous_resource_config(autonomous_resource)["resource"]["name"]
            #import pdb; pdb.set_trace()
            sample = result['results'][metric]['frames'][0]['data']['values']
            for timestamp, value in zip(sample[0], sample[1]):
                point = Point(measurement).tag("autonomous_resource", autonomous_resource_name).field(measurement, value).time(datetime.fromtimestamp(timestamp/1000))
                self.influx_write_api.write(self.influx_bucket, self.influx_org, point)
    
    def get_metrics_from_influxdb(self, measurement, autonomous_resource_name, time_range):
        query = f'from(bucket: "{self.influx_bucket}") |> range(start: {time_range}) |> filter(fn: (r) => r._measurement == "{measurement}" and r.autonomous_resource == "{autonomous_resource_name}")'
        tables = self.influx_query_api.query(org=self.influx_org, query=query)
        data = []
        for table in tables:
            for row in table.records:
              timestamp = pd.to_datetime(row.get_time())
              value = row.get_value()
              data.append([timestamp, value])
                          
        columns = ['timestamp', measurement]
        df = pd.DataFrame(data, columns=columns)

        return df
    
    def convert_to_seconds(self, frequency_str):
        seconds_per_unit = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}
        return int(frequency_str[:-1]) * seconds_per_unit[frequency_str[-1]]
    
    
def main():
    name = os.getenv("MONITOR_NAME")
    api_version = os.getenv("MONITOR_API_VERSION")
    kind = os.getenv("MONITOR_KIND")
    influx_url = os.getenv("INFLUX_URL")
    influx_token = os.getenv("INFLUX_TOKEN")
    influx_org = os.getenv("INFLUX_ORG")
    influx_bucket = os.getenv("INFLUX_BUCKET")

    monitor = Monitor(name, api_version, kind, influx_url, influx_token, influx_org, influx_bucket)

    actual_date = date.today()
    logging.basicConfig(handlers=[logging.StreamHandler()], level=logging.INFO,
                      format='%(asctime)s %(levelname)s %(message)s', datefmt='%m/%d/%Y %H:%M:%S')

    with ShutdownProtection(4) as protected_block:
        try:
            while True:
                autonomous_resources = monitor.get_autonomous_resources("dcna.dev/v1beta1")
                for ar in autonomous_resources.objects(monitor.kube_api):
                    frequency = monitor.convert_to_seconds(monitor.frequency)
                    logging.info(f"Collecting metrics from {ar.obj['spec']['resource']['name']}...")
                    monitor.save_metrics(monitor.collect_metrics_from_ds(ar), ar)
                    logging.info(f"Wating {frequency} seconds for the next run...")
                time.sleep(frequency)
        except (SystemExit, KeyboardInterrupt) as ex:
            print("Shutting down...")

if __name__ == "__main__":
    main()