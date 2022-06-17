#!/usr/bin/env python3
# Copyright (c) 2013-2020 GomSpace A/S. All rights reserved.

import time
import sys
import argparse
import json
import pymongo
from pymongo import MongoClient


def argumentsParser():
    parser = argparse.ArgumentParser(description='Export data from MongoDB')
    parser.add_argument('--host', dest="host", type=str,
                        default="10.0.8.91", help="MongoDB host name/ip")
    parser.add_argument('--db', dest="db", type=str,
                        default="test-adcs-db", help="MongoDB database name")
    parser.add_argument('--node', dest="node", type=int,
                        default="4", help='Override node from config file')
    parser.add_argument('--table', dest="table", type=int,
                        help='Override node from config file')
    parser.add_argument('--satellite', dest="satellite", type=int,
                        default="-1", help='Override satellite from config file')
    parser.add_argument('--from_ts', dest="from_ts", type=int,
                        default="-1", help='From timestamp')
    parser.add_argument('--to_ts', dest="to_ts", type=int,
                        default="-1", help='To timestamp')
    parser.add_argument('--config', dest="config", type=str,
                        default="config.json", help="Config JSON filename")
    parser.add_argument('--output', dest="output", type=str,
                        default="output.csv", help="Output CSV filename")
    parser.add_argument('--resolution', dest="resolution", type=int, default="1",
                        help='Resolution of timestamp (groups samples in e.g every 10 seconds)')
    parser.add_argument('--skip_incomplete', dest="skip_incomplete",
                        action="store_true", help='Skip incomplete rows')
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = argumentsParser()

    from_ts = args.from_ts
    to_ts = args.to_ts
    output_filename = args.output
    config_filename = args.config

    if from_ts == -1:
        from_ts = int(time.time()) - 2 * 3600
    if to_ts == -1:
        to_ts = int(time.time())

    # Database setup
    host = args.host
    port = int(27017)
    db_name = args.db
    collection = 'ParamData'

    # Set up host
    client = MongoClient(host, port)        # Server adress
    db = client[db_name]                # Database
    coll = db[collection]               # Collection

    try:
        with open(config_filename, "r") as fh:
            param_columns = json.load(fh)
    except Exception:
        print("Error opening config file {:s}".format(config_filename))
        raise

    # Count no columns
    no_columns = 0
    column_names = []
    column_dict = {}
    column_fmt_dict = {}
    for param_set in param_columns['parameters']:
        # print param_set
        for param in param_set['columns']:
            # print param
            param_name = list(param.keys())[0]
            fmt_str = ''
            alias_list = []
            if 'alias' in param[param_name]:
                alias_list = param[param_name]['alias']
            if 'fmt' in param[param_name]:
                fmt_str = param[param_name]['fmt']
            if 'index' in param[param_name]:
                no_columns += len(param[param_name]['index'])
                idx = 0
                for index in param[param_name]['index']:
                    name = param_name + '_' + str(index)
                    if len(alias_list):
                        column_names.append(alias_list[idx])
                    else:
                        column_names.append(name)
                    column_dict[name] = len(column_names)
                    column_fmt_dict[name] = fmt_str
                    idx += 1
            elif 'count' in param[param_name]:
                no_columns += int(param[param_name]['count'])
                idx = 0
                for index in range(0, param[param_name]['count']):
                    name = param_name + '_' + str(index)
                    if len(alias_list):
                        column_names.append(alias_list[idx])
                    else:
                        column_names.append(name)
                    column_dict[name] = len(column_names)
                    column_fmt_dict[name] = fmt_str
                    idx += 1
            else:
                no_columns += 1
                name = param_name
                if len(alias_list):
                    column_names.append(alias_list[0])
                else:
                    column_names.append(name)
                column_dict[name] = len(column_names)
                column_fmt_dict[name] = fmt_str
    print("Number of columns: {:d}".format(no_columns))
    # print column_names
    print(column_dict)
    print(column_fmt_dict)

    # Create dict to hold all timestamps
    ts_dict = {}
    # Set timestamp query
    ts_query = {"$gt": from_ts}
    if to_ts > from_ts:
        ts_query["$lt"] = to_ts
    for param_set in param_columns['parameters']:
        if 'timestamps' not in param_set:
            param_set['timestamps'] = 'epoch'
        for key in ['node', 'satellite']:
            if key not in param_set:
                print("\nWARNING: {0} not found in config\n".format(key))
                param_set[key] = -1
        print(param_set)

        node = args.node
        satellite = args.satellite
        table = args.table
        if node == -1:
            node = param_set['node']
        if satellite == -1:
            satellite = param_set['satellite']
        if node == -1:
            print("\nWARNING: Selector 'node' not specified in config\n")
        if satellite == -1:
            print("\nWARNING: Selector 'satellite' not specified in config\n")
        query_name = {'$in': []}
        for param in param_set['columns']:
            param_name = list(param.keys())[0]
            query_name['$in'].append(param_name)
        query = {"Param.Name": query_name, "Ts": ts_query}
        if node != -1:
            query["Param.Node"] = node
        if satellite != -1:
            query["Param.Satellite"] = satellite
        if table != None:
            query["Param.Table"] = table

        # Get doc from collection
        print('{:.3f}: Collecting data: {:s}'.format(time.time(), json.dumps(query)))

        param_list = list(coll.find(query))

        print("{:.3f}: Done: {:d}".format(time.time(), len(param_list)))
        for item in param_list:
            if args.resolution > 1:
                item['Ts'] = int(int(item['Ts']) /
                                 args.resolution) * args.resolution
            if not item['Ts'] in ts_dict:
                ts_dict[item['Ts']] = ['nan'] * (no_columns + 1)
                if param_set['timestamps'] == 'utc':
                    ts_dict[item['Ts']][0] = time.strftime(
                        "%Y.%m.%d %H:%M:%S", time.gmtime(int(item['Ts'])))
                elif param_set['timestamps'] == 'local':
                    ts_dict[item['Ts']][0] = time.strftime(
                        "%Y.%m.%d %H:%M:%S", time.localtime(int(item['Ts'])))
                else:
                    ts_dict[item['Ts']][0] = item['Ts']
            name = item['Param']['Name']
            if 'Index' in item['Param']:
                name += "_" + str(item['Param']['Index'])
            if name in column_dict:
                col = column_dict[name]  # + 1
                if column_fmt_dict[name] != '':
                    try:
                        ts_dict[item['Ts']][col] = column_fmt_dict[
                            name] % (item['Val'])
                    except Exception:
                        ts_dict[item['Ts']][col] = item['Val']
                else:
                    ts_dict[item['Ts']][col] = item['Val']

    print("Timestamps: {:d}".format(len(ts_dict.keys())))
    # print len(param_dict.keys())

    # Generate output in CSV format
    heading = "ts"
    for param_name in column_names:
        heading += ",%s" % (param_name)
    lines = 0
    written = 0
    try:
        print("Writing output file {:s}".format(output_filename))
        with open(output_filename, "w") as fh:
            fh.write(heading)
            fh.write("\n")
            for ts in sorted(ts_dict):
                lines += 1
                if args.skip_incomplete:
                    incomplete = False
                    for val in ts_dict[ts]:
                        if val == 'nan':
                            incomplete = True
                            break
                    if incomplete:
                        continue
                written += 1
                fh.write(",".join(str(x) for x in ts_dict[ts]))
                fh.write("\n")
    except Exception:
        print("Error opening {:s}".format(output_filename))
        raise

    print("Wrote {0} of {1} records to {2}".format(written, lines, output_filename))
