import datetime
import json
import urllib.request


NOCACHE = False

# 20-04-2019: server changed to https
SERVER_URL = "https://programmeren9.cmgt.hr.nl:9000/"

class Machine_Learning_Data():
        """ class om taining en test data op te halen en te sturen naar de server """

        def __init__(self, studentnummer=None):
                self.studentnummer = studentnummer

                assert type(studentnummer) is str, "Geef het studentnummer op als String"
                assert len(studentnummer) == 7 , "Geef (een correct) studentnummer op om de juiste data te krijgen"

        def get_data(self, url, cache=""):

                if cache != "" and not NOCACHE:
                        # probeer uit cache te halen en return
                        import os.path
                        if os.path.isfile(cache):
                            with open(cache) as cache_data:
                                return json.load(cache_data)

                req = urllib.request.Request(url)
                req.add_header('Accept', 'application/json')

                response = urllib.request.urlopen(req)

                # TODO: foutafhandeling
                data = json.loads(response.read().decode('utf8'))

                if cache != "":
                        # sla data op in cache
                        with open(cache, 'w') as fout:
                            json.dump(data, fout)

                return data

        def clustering_training(self):
                return self.get_data(SERVER_URL + self.studentnummer + "/clustering/training", self.studentnummer + "-clustering-training")

        def classification_training(self):
                return self.get_data(SERVER_URL + self.studentnummer + "/classification/training", self.studentnummer + "-classification-training")

        def classification_test(self, y=None):
                if y == None:
                        date_string = '{0:%Y-%m-%d}'.format(datetime.datetime.now())  # elke daga andere test-data
                        return self.get_data(SERVER_URL + self.studentnummer + "/classification/test", self.studentnummer + "-" + date_string + "-classification-test")
                else:
                    assert type(y) is list, "Stuur de classificaties als lijst"

                    data_y= json.dumps(y)
                    #print(data_y)

                    req = urllib.request.Request(SERVER_URL + self.studentnummer + "/classification/test",
                                                 data=data_y.encode('utf8'))
                    req.add_header('Content-Type', 'application/json')

                    response = urllib.request.urlopen(req)

                    # print(response)

                    return response.read().decode('utf8')




if __name__ == '__main__':
    data = Machine_Learning_Data("1234567")


    kmeans_training = data.kmeans_traing()

    for p in kmeans_training:
            print(p)