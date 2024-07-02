import requests
import json


class Datasource:
  def __init__(self, name, base_url ,access_token):
    self.name = name
    self.token = access_token
    self.base_url = base_url
    self.http_headers = {
        'Authorization': f'Bearer {self.token}',
        'Accept': 'application/json',
        'Content-Type': 'application/json'
    }
    self.info = self.get_info()


  def health(self):
    try:
      resp = requests.get(f"{self.base_url}/api/datasources/{self.info['id']}/health", headers=self.http_headers)
      resp.raise_for_status()
    except requests.exceptions.HTTPError as errh:
      print("HTTP Error")
      print(errh.args[0])
    if resp.json()['status'] == 'OK':
      return True

    return False


  def get_info(self):
    try:
      resp = requests.get(f'{self.base_url}/api/datasources/name/{self.name}', headers=self.http_headers)
      resp.raise_for_status()
    except requests.exceptions.HTTPError as errh:
      print("HTTP Error")
      print(errh.args[0])

    info = resp.json()

    return info


  def get_data(self, start_date, end_date, queries):
    grafana_queries = {
        "queries": queries,
        "from": end_date,
        "to": start_date
    }
    try:
      resp = requests.post(f"{self.base_url}/api/ds/query", headers=self.http_headers, data=json.dumps(grafana_queries))
      resp.raise_for_status()
    except requests.exceptions.HTTPError as errh:
      print("HTTP Error")
      print(errh.args[0])

    return resp.json()


class GrafanaQuery:
  def __init__(self, type, uid, refId):
      self.type = type
      self.uid = uid
      self.refId = refId


  def create_query(self, projectName, query_filters):

    query =         {
          "datasource": {
            "type": self.type,
            "uid": self.uid
          },
          "queryType": "timeSeriesList",
          "refId": self.refId,
          "timeSeriesList": {
            "alignmentPeriod": "+10s",
            "crossSeriesReducer": "REDUCE_NONE",
            "filters": query_filters,
            "groupBys": [],
            "perSeriesAligner": "ALIGN_NONE",
            "preprocessor": "none",
            "projectName": projectName,
            "view": "FULL"
          },
          "intervalMs": 10000,
          "datasourceId": 15,
          "maxDataPoints": 1904
        }
    return query