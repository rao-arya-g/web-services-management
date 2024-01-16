from pymongo import MongoClient
import bson
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

    def keyword_search(self, collection, search_string):
        list_of_keywords = search_string.split(",")
        list_of_keywords = [x.strip() for x in list_of_keywords]
        collection_obj = self.db.get_collection(collection)

        outer_and_conditions = []
        for keyword in list_of_keywords:
            regex_exp = bson.regex.Regex(r"(?i)\b{}\b".format(keyword))
            or_condition = {"$or": [{"title": regex_exp}, {"summary": regex_exp}, {"description": regex_exp}]}
            outer_and_conditions.append(or_condition)

        filtered_results = collection_obj.find({"$and": outer_and_conditions})
        output_dict = {}
        for entry in filtered_results:
            filtered_entry = dict(entry)
            filtered_entry.pop('_id')
            output_dict[filtered_entry['id']] = filtered_entry

        if len(output_dict) == 0:
            return pd.DataFrame(columns=["id", "name", "title", "summary", "label"])

        data_df = pd.DataFrame.from_dict(output_dict, orient="index")
        data_df = data_df.reset_index(drop=True)
        data_df = data_df[["id", "name", "title", "summary", "label"]]
        return data_df

    def search_apis_by_keywords(self, search_string):
        return self.keyword_search("API_DATA", search_string)

    def search_mashup_by_keywords(self, search_string):
        return self.keyword_search("MASHUP_DATA", search_string)

    @staticmethod
    def get_comparison_operator(input_string):
        if ">=" in input_string:
            return "$gte", float(input_string[2:])
        elif "<=" in input_string:
            return "$lte", float(input_string[2:])
        elif ">" in input_string:
            return "$gt", float(input_string[1:])
        elif "<" in input_string:
            return "$lt", float(input_string[1:])
        else:
            return "$eq", float(input_string)

    def criteria_search(self, collection_name, query_params):

        query_params = dict(query_params)
        filter_dict = []
        for param in query_params:

            if param in ["rating"] and query_params[param] != "":
                try:
                    operator, value = self.get_comparison_operator(query_params[param])
                    filter_dict.append({"rating": {operator: value}})
                except Exception as _:
                    continue

            elif param in ["Tags", "tags"] and query_params[param] != "":
                try:
                    if "," in query_params[param]:
                        values = [bson.regex.Regex(r"(?i)\b{}\b".format(x.strip())) for x in query_params[param].split(',')]
                        filter_dict.append({param: {"$all": values}})
                    else:
                        regex_exp = bson.regex.Regex(r"(?i)\b{}\b".format(query_params[param]))
                        filter_dict.append({param: regex_exp})

                except Exception as _:
                    continue

            elif param in ["updated", "APIs"] and query_params[param] != "":
                regex_exp = bson.regex.Regex(r"^{}".format(query_params[param]))
                filter_dict.append({param: regex_exp})

            elif query_params[param] is not None and query_params[param] != "":
                filter_dict.append({param: query_params[param]})

        collection_obj = self.db.get_collection(collection_name)
        filtered_results = collection_obj.find({"$and": filter_dict})

        output_dict = {}
        for entry in filtered_results:
            filtered_entry = dict(entry)
            filtered_entry.pop('_id')
            output_dict[filtered_entry['id']] = filtered_entry

        if len(output_dict) == 0:
            return pd.DataFrame(columns=["id", "name", "title", "summary", "label"])

        data_df = pd.DataFrame.from_dict(output_dict, orient="index")
        data_df = data_df.reset_index(drop=True)
        data_df = data_df[["id", "name", "title", "summary", "label"]]
        return data_df

    def search_apis_by_criteria(self, query_params):
        return self.criteria_search("API_DATA", query_params)

    def search_mashup_by_criteria(self, query_params):
        return self.criteria_search("MASHUP_DATA", query_params)


if __name__ == '__main__':
    db_obj = DatabaseWrapper()

    random_output = db_obj.search_mashup_by_criteria({"APIs": "Flickr"})
    print(random_output)
