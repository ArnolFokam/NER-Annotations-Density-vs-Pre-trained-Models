DATA_DICT = {
    'global_cap_sentences': {0.01: {'amh': 0.008900756564307966,
                                 'conll_2003_en': 0.010418805173475673,
                                 'hau': 0.008774128854349462,
                                 'ibo': 0.008604794099569761,
                                 'kin': 0.00606227610912097,
                                 'lug': 0.015284552845528456,
                                 'luo': 0.009666080843585237,
                                 'pcm': 0.008635578583765112,
                                 'swa': 0.013566739606126914,
                                 'wol': 0.0136986301369863,
                                 'yor': 0.012694958287994197},
                          0.05: {'amh': 0.04984423676012461,
                                 'conll_2003_en': 0.048655306918497225,
                                 'hau': 0.04637753823013287,
                                 'ibo': 0.05009219422249539,
                                 'kin': 0.040782584734086524,
                                 'lug': 0.05040650406504065,
                                 'luo': 0.0492091388400703,
                                 'pcm': 0.045768566493955096,
                                 'swa': 0.04463894967177243,
                                 'wol': 0.0410958904109589,
                                 'yor': 0.047515415306492566},
                          0.1: {'amh': 0.10858923008455719,
                                'conll_2003_en': 0.10577910080065694,
                                'hau': 0.09902231135622963,
                                'ibo': 0.09987707437000615,
                                'kin': 0.10112978782033619,
                                'lug': 0.0991869918699187,
                                'luo': 0.10896309314586995,
                                'pcm': 0.09995682210708118,
                                'swa': 0.11181619256017505,
                                'wol': 0.1110310021629416,
                                'yor': 0.1048240841494378},
                          0.2: {'amh': 0.17935024477080552,
                                'conll_2003_en': 0.20047218230342845,
                                'hau': 0.19704186512910504,
                                'ibo': 0.1972956361401352,
                                'kin': 0.19454395150179113,
                                'lug': 0.20650406504065041,
                                'luo': 0.19420035149384884,
                                'pcm': 0.16342832469775476,
                                'swa': 0.19649890590809627,
                                'wol': 0.19610670511896178,
                                'yor': 0.19477693144722524},
                          0.3: {'amh': 0.27547841566533154,
                                'conll_2003_en': 0.3014781359063847,
                                'hau': 0.2985710704437202,
                                'ibo': 0.32329440688383526,
                                'kin': 0.3069716175254891,
                                'lug': 0.3047154471544715,
                                'luo': 0.27855887521968364,
                                'pcm': 0.3035405872193437,
                                'swa': 0.31159737417943106,
                                'wol': 0.2869502523431867,
                                'yor': 0.30395357272397533},
                          0.4: {'amh': 0.38896306186025814,
                                'conll_2003_en': 0.3956066516115787,
                                'hau': 0.3940837302582101,
                                'ibo': 0.4065765212046712,
                                'kin': 0.37779002480022045,
                                'lug': 0.4097560975609756,
                                'luo': 0.39630931458699475,
                                'pcm': 0.41278065630397237,
                                'swa': 0.3903719912472648,
                                'wol': 0.4030281182408075,
                                'yor': 0.40841494377947046},
                          0.5: {'amh': 0.514018691588785,
                                'conll_2003_en': 0.4968692260316157,
                                'hau': 0.4845826021559288,
                                'ibo': 0.4963122311001844,
                                'kin': 0.5023422430421604,
                                'lug': 0.4835772357723577,
                                'luo': 0.4938488576449912,
                                'pcm': 0.5338946459412781,
                                'swa': 0.5164113785557987,
                                'wol': 0.5155010814708003,
                                'yor': 0.5302865433442148},
                          0.6: {'amh': 0.5950155763239875,
                                'conll_2003_en': 0.6003387394785465,
                                'hau': 0.6001504136375031,
                                'ibo': 0.5774431468961279,
                                'kin': 0.6048498208872968,
                                'lug': 0.6133333333333333,
                                'luo': 0.6089630931458699,
                                'pcm': 0.5848445595854922,
                                'swa': 0.5954048140043764,
                                'wol': 0.5912040374909877,
                                'yor': 0.6010155966630395},
                          0.7: {'amh': 0.7125055629728527,
                                'conll_2003_en': 0.7021145555327448,
                                'hau': 0.7056906492855353,
                                'ibo': 0.6982175783650891,
                                'kin': 0.7068062827225131,
                                'lug': 0.7180487804878048,
                                'luo': 0.70298769771529,
                                'pcm': 0.7141623488773747,
                                'swa': 0.6757111597374179,
                                'wol': 0.7007930785868781,
                                'yor': 0.7065651070003627},
                          0.8: {'amh': 0.8010680907877169,
                                'conll_2003_en': 0.7993738452063232,
                                'hau': 0.8059664076209576,
                                'ibo': 0.8103872157344807,
                                'kin': 0.8128961146321301,
                                'lug': 0.7869918699186992,
                                'luo': 0.7820738137082601,
                                'pcm': 0.7953367875647669,
                                'swa': 0.8030634573304157,
                                'wol': 0.7945205479452054,
                                'yor': 0.8186434530286544},
                          0.9: {'amh': 0.9034267912772586,
                                'conll_2003_en': 0.9015089304044344,
                                'hau': 0.8989721734770619,
                                'ibo': 0.9145666871542717,
                                'kin': 0.8994213281895839,
                                'lug': 0.8985365853658537,
                                'luo': 0.9068541300527241,
                                'pcm': 0.9060880829015544,
                                'swa': 0.8975929978118162,
                                'wol': 0.9062725306416727,
                                'yor': 0.9027928908233587},
                          1.0: {'amh': 1.0,
                                'conll_2003_en': 1.0,
                                'hau': 1.0,
                                'ibo': 1.0,
                                'kin': 1.0,
                                'lug': 1.0,
                                'luo': 1.0,
                                'pcm': 1.0,
                                'swa': 1.0,
                                'wol': 1.0,
                                'yor': 1.0}},
 'local_cap_labels': {1: {'amh': 0.536270582999555,
                          'conll_2003_en': 0.5121125025662082,
                          'hau': 0.3848082226121835,
                          'ibo': 0.4658881376767056,
                          'kin': 0.41361256544502617,
                          'lug': 0.36,
                          'luo': 0.44200351493848855,
                          'pcm': 0.3331174438687392,
                          'swa': 0.36433260393873085,
                          'wol': 0.5537130497476568,
                          'yor': 0.44940152339499456},
                      2: {'amh': 0.82688028482421,
                          'conll_2003_en': 0.7825908437692466,
                          'hau': 0.6467786412634745,
                          'ibo': 0.7387830362630609,
                          'kin': 0.6718104160925875,
                          'lug': 0.6097560975609756,
                          'luo': 0.7240773286467487,
                          'pcm': 0.564119170984456,
                          'swa': 0.6225382932166302,
                          'wol': 0.802451333813987,
                          'yor': 0.7138193688792165},
                      3: {'amh': 0.9488206497552292,
                          'conll_2003_en': 0.9037158694313283,
                          'hau': 0.8094760591626974,
                          'ibo': 0.8847572218807621,
                          'kin': 0.8255717828602921,
                          'lug': 0.7596747967479675,
                          'luo': 0.8646748681898067,
                          'pcm': 0.6999136442141624,
                          'swa': 0.7923413566739607,
                          'wol': 0.9120403749098774,
                          'yor': 0.8599927457381211},
                      4: {'amh': 0.9857587894971073,
                          'conll_2003_en': 0.9646376514062821,
                          'hau': 0.9007269992479318,
                          'ibo': 0.9575906576521205,
                          'kin': 0.906034720308625,
                          'lug': 0.8419512195121951,
                          'luo': 0.9411247803163445,
                          'pcm': 0.7780656303972366,
                          'swa': 0.8910284463894967,
                          'wol': 0.9596250901225667,
                          'yor': 0.931809938338774},
                      5: {'amh': 0.9968847352024922,
                          'conll_2003_en': 0.9824984602750975,
                          'hau': 0.9501128102281273,
                          'ibo': 0.9830977258758451,
                          'kin': 0.9468173050427114,
                          'lug': 0.896260162601626,
                          'luo': 0.9771528998242531,
                          'pcm': 0.8188687392055267,
                          'swa': 0.9468271334792122,
                          'wol': 0.9790915645277577,
                          'yor': 0.9655422560754443},
                      6: {'amh': 1.0,
                          'conll_2003_en': 0.9902997331143503,
                          'hau': 0.9741789922286287,
                          'ibo': 0.9917025199754149,
                          'kin': 0.9699641774593551,
                          'lug': 0.9255284552845529,
                          'luo': 0.9885764499121266,
                          'pcm': 0.8426165803108808,
                          'swa': 0.973741794310722,
                          'wol': 0.9884643114635905,
                          'yor': 0.9836779107725789},
                      7: {'amh': 1.0,
                          'conll_2003_en': 0.9938924245534798,
                          'hau': 0.986964151416395,
                          'ibo': 0.9960049170251998,
                          'kin': 0.9820887296775971,
                          'lug': 0.9421138211382114,
                          'luo': 0.9947275922671354,
                          'pcm': 0.8570811744386874,
                          'swa': 0.9857768052516411,
                          'wol': 0.9935111751982696,
                          'yor': 0.9923830250272034},
                      8: {'amh': 1.0,
                          'conll_2003_en': 0.9956374461096285,
                          'hau': 0.9934820757081976,
                          'ibo': 0.9978488014751076,
                          'kin': 0.9881510057867181,
                          'lug': 0.9541463414634146,
                          'luo': 0.9973637961335676,
                          'pcm': 0.8670120898100173,
                          'swa': 0.9923413566739606,
                          'wol': 0.9949531362653208,
                          'yor': 0.9956474428726877},
                      9: {'amh': 1.0,
                          'conll_2003_en': 0.996612605214535,
                          'hau': 0.9967410378540987,
                          'ibo': 0.9987707437000615,
                          'kin': 0.9925599338660788,
                          'lug': 0.9619512195121951,
                          'luo': 0.9982425307557118,
                          'pcm': 0.8752158894645942,
                          'swa': 0.9960612691466083,
                          'wol': 0.996395097332372,
                          'yor': 0.9974610083424011},
                      10: {'amh': 1.0,
                           'conll_2003_en': 0.9974337918291932,
                           'hau': 0.9984958636249687,
                           'ibo': 0.9996926859250154,
                           'kin': 0.9950399559107193,
                           'lug': 0.9678048780487805,
                           'luo': 0.9991212653778558,
                           'pcm': 0.8827720207253886,
                           'swa': 0.9978118161925602,
                           'wol': 0.9971160778658976,
                           'yor': 0.9981864345302865}},
 'local_swap_labels_like_cap': {1: {'amh': 0.536270582999555,
                                    'conll_2003_en': 0.5121125025662082,
                                    'hau': 0.3848082226121835,
                                    'ibo': 0.4658881376767056,
                                    'kin': 0.41361256544502617,
                                    'lug': 0.36,
                                    'luo': 0.44200351493848855,
                                    'pcm': 0.3331174438687392,
                                    'swa': 0.36433260393873085,
                                    'wol': 0.5537130497476568,
                                    'yor': 0.44940152339499456},
                                2: {'amh': 0.82688028482421,
                                    'conll_2003_en': 0.7825908437692466,
                                    'hau': 0.6467786412634745,
                                    'ibo': 0.7387830362630609,
                                    'kin': 0.6718104160925875,
                                    'lug': 0.6097560975609756,
                                    'luo': 0.7240773286467487,
                                    'pcm': 0.564119170984456,
                                    'swa': 0.6225382932166302,
                                    'wol': 0.802451333813987,
                                    'yor': 0.7138193688792165},
                                3: {'amh': 0.9488206497552292,
                                    'conll_2003_en': 0.9037158694313283,
                                    'hau': 0.8094760591626974,
                                    'ibo': 0.8847572218807621,
                                    'kin': 0.8255717828602921,
                                    'lug': 0.7596747967479675,
                                    'luo': 0.8646748681898067,
                                    'pcm': 0.6999136442141624,
                                    'swa': 0.7923413566739607,
                                    'wol': 0.9120403749098774,
                                    'yor': 0.8599927457381211},
                                4: {'amh': 0.9857587894971073,
                                    'conll_2003_en': 0.9646376514062821,
                                    'hau': 0.9007269992479318,
                                    'ibo': 0.9575906576521205,
                                    'kin': 0.906034720308625,
                                    'lug': 0.8419512195121951,
                                    'luo': 0.9411247803163445,
                                    'pcm': 0.7780656303972366,
                                    'swa': 0.8910284463894967,
                                    'wol': 0.9596250901225667,
                                    'yor': 0.931809938338774},
                                5: {'amh': 0.9968847352024922,
                                    'conll_2003_en': 0.9824984602750975,
                                    'hau': 0.9501128102281273,
                                    'ibo': 0.9830977258758451,
                                    'kin': 0.9468173050427114,
                                    'lug': 0.896260162601626,
                                    'luo': 0.9771528998242531,
                                    'pcm': 0.8188687392055267,
                                    'swa': 0.9468271334792122,
                                    'wol': 0.9790915645277577,
                                    'yor': 0.9655422560754443},
                                6: {'amh': 1.0,
                                    'conll_2003_en': 0.9902997331143503,
                                    'hau': 0.9741789922286287,
                                    'ibo': 0.9917025199754149,
                                    'kin': 0.9699641774593551,
                                    'lug': 0.9255284552845529,
                                    'luo': 0.9885764499121266,
                                    'pcm': 0.8426165803108808,
                                    'swa': 0.973741794310722,
                                    'wol': 0.9884643114635905,
                                    'yor': 0.9836779107725789},
                                7: {'amh': 1.0,
                                    'conll_2003_en': 0.9938924245534798,
                                    'hau': 0.986964151416395,
                                    'ibo': 0.9960049170251998,
                                    'kin': 0.9820887296775971,
                                    'lug': 0.9421138211382114,
                                    'luo': 0.9947275922671354,
                                    'pcm': 0.8570811744386874,
                                    'swa': 0.9857768052516411,
                                    'wol': 0.9935111751982696,
                                    'yor': 0.9923830250272034},
                                8: {'amh': 1.0,
                                    'conll_2003_en': 0.9956374461096285,
                                    'hau': 0.9934820757081976,
                                    'ibo': 0.9978488014751076,
                                    'kin': 0.9881510057867181,
                                    'lug': 0.9541463414634146,
                                    'luo': 0.9973637961335676,
                                    'pcm': 0.8670120898100173,
                                    'swa': 0.9923413566739606,
                                    'wol': 0.9949531362653208,
                                    'yor': 0.9956474428726877},
                                9: {'amh': 1.0,
                                    'conll_2003_en': 0.996612605214535,
                                    'hau': 0.9967410378540987,
                                    'ibo': 0.9987707437000615,
                                    'kin': 0.9925599338660788,
                                    'lug': 0.9619512195121951,
                                    'luo': 0.9982425307557118,
                                    'pcm': 0.8752158894645942,
                                    'swa': 0.9960612691466083,
                                    'wol': 0.996395097332372,
                                    'yor': 0.9974610083424011},
                                10: {'amh': 1.0,
                                     'conll_2003_en': 0.9974337918291932,
                                     'hau': 0.9984958636249687,
                                     'ibo': 0.9996926859250154,
                                     'kin': 0.9950399559107193,
                                     'lug': 0.9678048780487805,
                                     'luo': 0.9991212653778558,
                                     'pcm': 0.8827720207253886,
                                     'swa': 0.9978118161925602,
                                     'wol': 0.9971160778658976,
                                     'yor': 0.9981864345302865}}}