from pymongo import MongoClient
import pandas as pd


class DatabaseWrapper(object):
    connection = None
    hostname = "localhost"
    port_number = 27017
    database_name = "SERVICE_DATA"
    list_of_connections = ["API_DATA", "MASHUP_DATA"]

    def __init__(self):
        if DatabaseWrapper.connection is None:
            DatabaseWrapper.connection = MongoClient(DatabaseWrapper.hostname, DatabaseWrapper.port_number)

        self.connection = DatabaseWrapper.connection

        list_of_databases = self.connection.list_database_names()
        if DatabaseWrapper.database_name not in list_of_databases:
            print("Creating a new DB instance with Database name {0}".format(self.database_name))
            self.db = self.connection[DatabaseWrapper.database_name]
        else:
            print("DB {0} already exists. Not creating a DB instance".format(self.database_name))
            self.db = self.connection.get_database(DatabaseWrapper.database_name)

        self.add_collection_to_db()

    def add_collection_to_db(self):
        all_existing_collection = self.db.list_collection_names()
        for collection in DatabaseWrapper.list_of_connections:
            if collection not in all_existing_collection:
                print("Creating a new collection with name {0}".format(collection))
                self.db.create_collection(collection)
            else:
                print("Collection {0} already exists. Not creating a new collection".format(collection))

    def get_web_apis(self):
        output_dict = {}
        collection_obj = self.db.get_collection("API_DATA")

        filtered_results = list(collection_obj.find({}))

        for entry in filtered_results:
            filtered_entry = dict(entry)
            filtered_entry.pop('_id')
            output_dict[filtered_entry['id']] = filtered_entry

        data_df = pd.DataFrame.from_dict(output_dict, orient="index")
        data_df = data_df.reset_index(drop=True)
        data_df = data_df[["id", "title", "name", "summary", "label", "description", "Tags", "category"]]
        return data_df

    def get_web_mashups(self):
        output_dict = {}
        collection_obj = self.db.get_collection("MASHUP_DATA")

        filtered_results = list(collection_obj.find({}))

        for entry in filtered_results:
            filtered_entry = dict(entry)
            filtered_entry.pop('_id')
            output_dict[filtered_entry['id']] = filtered_entry

        data_df = pd.DataFrame.from_dict(output_dict, orient="index")
        data_df = data_df.reset_index(drop=True)
        data_df = data_df[["id", "title", "name", "summary", "label", "description", "tags"]]
        return data_df


if __name__ == '__main__':
    db_obj = DatabaseWrapper()

    x = db_obj.get_web_apis()
    x.to_clipboard()
