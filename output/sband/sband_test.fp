sband0_1,rparam download 4 53                               ,0,0,0,0,1629450600,0,0
sband0_2,rparam set gso_ant 0.0000 0.0000 0.0000            ,0,0,0,0,1629450601,0,0
sband0_3,rparam set en_gso 1 0 1                            ,0,0,0,0,1629450602,0,0
sband0_4,rparam set gso_list 0.0 0.0 0.0                    ,0,0,0,0,1629450603,0,0
sband0_5,rparam send                                        ,0,0,0,0,1629450604,0,0
sband0_6,adcs server 4 20                                   ,0,0,0,0,1629450605,0,0
sband0_7,adcs state setacs 6                                ,0,0,0,0,1629450606,0,0
sband0_8,power on 9 6 0 900                                 ,0,0,0,0,1629451680,0,0
sband0_9,cmp clock sync 2                                   ,0,0,0,0,1629451711,0,0
sband0_10,power on 9 1 0 850                                ,0,0,0,0,1629451722,0,0
sband0_11,gscript server 2                                  ,0,0,0,0,1629451753,0,0
sband0_12,gscript run /data/scripts/sband_sat_enable.gsh    ,0,0,0,0,1629451754,0,0
sband0_13,z7000 node 2                                      ,0,0,0,0,1629452400,0,0
sband0_14,z7000 cmd exec /sbin/shutdown shutdown now        ,0,0,0,0,1629452401,0,0
sband0_15,power off 9 1                                     ,0,0,0,0,1629452432,0,0
sband0_16,power off 9 6                                     ,0,0,0,0,1629452463,0,0
sband0_17,rparam download 4 53                              ,0,0,0,0,1629452494,0,0
sband0_18,rparam set euler_offset 0.0000 0.0000 0.0000      ,0,0,0,0,1629452495,0,0
sband0_19,rparam set en_gso 0 0 1                           ,0,0,0,0,1629452496,0,0
sband0_20,rparam send                                       ,0,0,0,0,1629452497,0,0
sband0_21,adcs server 4 20                                  ,0,0,0,0,1629452498,0,0
sband0_22,adcs state setacs 6                               ,0,0,0,0,1629452499,0,0
sband0_23,hk server 1                                       ,0,0,0,0,1629452501,0,0
sband0_24,hk get 25 10 85 1629452500 /flash1/sband25_0.bin  ,0,0,0,0,1629452502,0,0
sband0_25,ftp server 1                                      ,0,0,0,0,1629452534,0,0
sband0_26,ftp zip /flash1/sband25_0.bin /flash1/sband25_0.zip,0,0,0,0,1629452535,0,0
sband0_27,hk server 1                                       ,0,0,0,0,1629452567,0,0
sband0_28,hk get 22 10 85 1629452500 /flash1/sband22_0.bin  ,0,0,0,0,1629452568,0,0
sband0_29,ftp server 1                                      ,0,0,0,0,1629452600,0,0
sband0_30,ftp zip /flash1/sband22_0.bin /flash1/sband22_0.zip,0,0,0,0,1629452601,0,0
sband0_31,hk server 1                                       ,0,0,0,0,1629452633,0,0
sband0_32,hk get 23 10 85 1629452500 /flash1/sband23_0.bin  ,0,0,0,0,1629452634,0,0
sband0_33,ftp server 1                                      ,0,0,0,0,1629452666,0,0
sband0_34,ftp zip /flash1/sband23_0.bin /flash1/sband23_0.zip,0,0,0,0,1629452667,0,0
sband0_35,hk_srv beacon samplerate 25 high                  ,0,0,0,0,1629452698,0,0
