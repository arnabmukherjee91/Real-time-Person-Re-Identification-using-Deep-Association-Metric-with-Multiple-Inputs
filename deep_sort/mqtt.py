import paho.mqtt.client as mqtt
import numpy as np
import time
import json


temp = {}

def on_message(client, userdata, message):
	global temp
	receive = json.loads(message.payload.decode('utf-8'))
	temp = receive.copy()
	print(temp)

class NumpyEncoder(json.JSONEncoder):
	def default(self, obj):
		if isinstance(obj, np.ndarray):
			return obj.tolist()
		return json.JSONEncoder.default(self, obj)

def deliver(mean,covariance,track_id,hits,age,time_since_update,state,features,n_init,max_age):
	data = {track_id:[mean,covariance,hits,age,time_since_update,state,features,n_init,max_age]}
	output = json.dumps(data, cls=NumpyEncoder)
	#print("deliver : " + output)
	client.publish(topic="TestingTopic2", payload=output, qos=0, retain=False)
'''
def deliver(mean,covariance,track_id,hits,age,time_since_update,state,features,n_init,max_age):
	data = {mean,covariance,track_id,hits,age,time_since_update,state,features,n_init,max_age}
	output = json.dumps(data, cls=NumpyEncoder)
	#print("deliver : " + output)
	client.publish(topic="TestingTopic2", payload=output, qos=0, retain=False)
'''

broker_url = "mqtt.eclipse.org"
broker_port = 1883

client = mqtt.Client()
client.connect(broker_url, broker_port)
client.on_message = on_message
client.subscribe("TestingTopic", qos=1)
client.loop_start()
