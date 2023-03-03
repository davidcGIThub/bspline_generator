"""
This module contains code that converts b-spline control points
to Bezier curve control points
"""
import numpy as np
from bsplinegenerator.helper_functions import count_number_of_control_points

def convert_to_minvo_control_points(bspline_control_points):
    number_of_control_points = count_number_of_control_points(bspline_control_points)
    order = number_of_control_points - 1
    if order > 7:
        raise Exception("Can only retrieve conversion matrix for curves of order 1-7")
    conversion_matrix = get_bspline_to_minvo_conversion_matrix(order)
    minvo_control_points = np.transpose(np.dot(conversion_matrix, np.transpose(bspline_control_points)))
    return minvo_control_points

def convert_list_to_minvo_control_points(bspline_control_points,order):
    number_of_bspline_control_points = count_number_of_control_points(bspline_control_points)
    composite_conversion_matrix = get_composite_bspline_to_minvo_conversion_matrix(number_of_bspline_control_points,order)
    minvo_control_point_list = np.transpose(np.dot(composite_conversion_matrix, np.transpose(bspline_control_points)))
    return minvo_control_point_list

def get_composite_bspline_to_minvo_conversion_matrix(num_bspline_control_points, order):
    if order > 7:
        raise Exception("Can only retrieve conversion matrix for curves of order 1-7")
    number_of_knot_point_segments = num_bspline_control_points - order
    number_of_minvo_control_points = (number_of_knot_point_segments)*(order+1)
    composite_conversion_matrix = np.zeros((number_of_minvo_control_points,num_bspline_control_points))
    conversion_matrix = get_bspline_to_minvo_conversion_matrix(order)
    for i in range(number_of_knot_point_segments):
        composite_conversion_matrix[i*(order+1):i*(order+1)+order+1 , i:i+order+1] = conversion_matrix
    return composite_conversion_matrix

def get_bspline_to_minvo_conversion_matrix(order):
    conversion_matrix = np.array([])

    if order == 1:
        conversion_matrix = np.array([[1, 0],
                                      [0, 1]])
    elif order == 2:
        conversion_matrix = np.array([[ 0.53867513,  0.08333333, -0.03867513],
                                      [ 0.5,         0.83333333 , 0.5       ],
                                      [-0.03867513 , 0.08333333 , 0.53867513]]).T
    elif order == 3:
        # conversion_matrix = np.array([[   1/6,  0.057009542139797595613306102386893, -0.015455156825262485566573649098775, 0],
        # [    2/3,   0.66657381574108923111064205020873,    0.2918717989443756838876956809183,    1/6],
        # [   1/6,    0.2918717989443756838876956809183,   0.66657381574108923111064205020873,     2/3],
        # [0, -0.015455156825262485566573649098775,  0.057009542139797595613306102386893,    1/6]]).T
        conversion_matrix = np.array([[   0.18372189964688778830269864557208,  0.057009542139797595613306102386893, -0.015455156825262485566573649098775, -0.0053387946850481119404479942697845],
        [    0.7017652268843997637057156686535,   0.66657381574108923111064205020873,    0.2918717989443756838876956809183,    0.11985166815376058497710386445935],
        [   0.11985166815376058497710386445935,    0.2918717989443756838876956809183,   0.66657381574108923111064205020873,     0.7017652268843997637057156686535],
        [-0.0053387946850481119404479942697845, -0.015455156825262485566573649098775,  0.057009542139797595613306102386893,    0.18372189964688778830269864557208]]).T

        
        
    elif order == 4:
        conversion_matrix = np.array([[ 0.04654488,  0.01902005, -0.0020279,  -0.00193449, -0.00105254],
                                      [ 0.49370783,  0.40331371,  0.18162853,  0.0395948,   0.01773369],
                                      [ 0.44306614,  0.54000593, 0.64079874,  0.54000593,  0.44306614],
                                      [ 0.01773369,  0.0395948,   0.18162853,  0.40331371,  0.49370783],
                                      [-0.00105254, -0.00193449, -0.0020279,   0.01902005,  0.04654488]]).T
    elif order == 5:
        conversion_matrix = np.array([[ 9.36633383e-03,  4.59221748e-03, -2.41070259e-05, -3.64703996e-04, -3.05969926e-04, -1.39856086e-04],
                                        [ 2.35803353e-01,  1.88464610e-01,  8.68737506e-02 , 2.46973953e-02, -6.38854070e-04,  2.36477513e-04],
                                        [ 5.64969670e-01,  5.74893473e-01,  5.12739907e-01,  3.76077758e-01, 2.32994524e-01,  1.89764022e-01],
                                        [ 1.89764022e-01 , 2.32994524e-01 , 3.76077758e-01 , 5.12739907e-01, 5.74893473e-01,  5.64969670e-01],
                                        [ 2.36477513e-04 ,-6.38854070e-04 , 2.46973953e-02 , 8.68737506e-02, 1.88464610e-01 , 2.35803353e-01],
                                        [-1.39856086e-04 ,-3.05969926e-04, -3.64703996e-04, -2.41070259e-05, 4.59221748e-03,  9.36633383e-03]]).T
    elif order == 6:
        conversion_matrix = np.array([[0.0015673013261596200063691627234054,   0.00086365958573481459082934164718087,  0.000072529009510193981453523389713465, -0.000060827300198005163150132104527589, -0.000048617373859850341236426744525371, -0.000031602487218597667419605402957684, -0.000019836374100934917533108312851288],
                [ 0.086685494064457982661616041913735,     0.069771729463297254362491229999255,     0.034459522555511007351613681598858,     0.011017789796016227347952001966402,    0.001022874621471957088779731829266,   -0.0017069297382149851628753259873781, -0.00089356992228718492515564483879434],
                [ 0.44151053682001590804989716687095,      0.42058964124015786726787258966978,      0.33399052351722979509038395738409,      0.2241835255776350962041341071433,      0.12932400505749663927837287343786,     0.070868392911277499117966039958345,   0.061201150391309814097279671094249],
                [ 0.4099489236944430352846165804626,       0.43964510902496538718275408954472,      0.50117916261264040485981047665333,      0.5297190238530935108259472436798,      0.50117916261264040485981047665333,     0.43964510902496538718275408954472,    0.4099489236944430352846165804626],
                [ 0.061201150391309814097279671094249,     0.070868392911277499117966039958345,     0.12932400505749663927837287343786,      0.2241835255776350962041341071433,      0.33399052351722979509038395738409,     0.42058964124015786726787258966978,    0.44151053682001590804989716687095],
                [-0.00089356992228718492515564483879434,  -0.0017069297382149851628753259873781,    0.001022874621471957088779731829266,     0.011017789796016227347952001966402,    0.034459522555511007351613681598858,    0.069771729463297254362491229999255,   0.086685494064457982661616041913735],
                [-0.000019836374100934917533108312851288, -0.000031602487218597667419605402957684, -0.000048617373859850341236426744525371, -0.000060827300198005163150132104527589, 0.000072529009510193981453523389713465, 0.00086365958573481459082934164718087, 0.0015673013261596200063691627234054]]).T
    
    elif order == 7:
        conversion_matrix = np.array([[0.00022428031175139061285260626677095,    0.00013513874256402826186596446559443,   0.000020709635337561553818374320063958,  -0.000006886557963029641619274086529499, -0.0000077356098441786850369139886516143, -0.0000039355521071470918751517887926294, -0.0000040323323856154789413751338051599, -0.0000021869677709226337744440535293076],
                [0.026142355417612899750687673989683,       0.02148526549159615482191328772425,      0.011240277531447318192462756951911,     0.0041848756747159772795357862370816,    0.00058803234003523545146332593303603,   -0.00029378157094206690131902596749094,   -0.00064742428750239711695222490399308,   -0.00033092823209790038775099705535793],
                [0.2516484495603468383732647824107,       0.23472600257366537822644304610283,       0.17697599487580784859160458888522,        0.1161643441467228179916394850984,      0.063070703692555895654085010625911,      0.032951859810865562283675061080646,      0.015643714659805141795624074118701,      0.015590965226372579133576396131442],
                [0.48780380655924084588080960092844,       0.49433520626075415410048915345936,        0.4821947826777955353541460146772,       0.44395414677862366884378260553705,       0.37205251953515363857856817991431,       0.29691409259179551833419611396976,       0.23432612889150319883642357199768,       0.21892325812454416218463854732815],
                [0.21892325812454416218463854732815,       0.23432612889150319883642357199768,       0.29691409259179551833419611396976,       0.37205251953515363857856817991431,       0.44395414677862366884378260553705,        0.4821947826777955353541460146772,       0.49433520626075415410048915345936,       0.48780380655924084588080960092844],
                [0.015590965226372579167149098849076,      0.015643714659805141857050041081942,      0.032951859810865562352497242188625,      0.063070703692555895733355465791096,       0.11616434414672281786163527853815,       0.17697599487580784786961919759341,       0.23472600257366537651617348525876,       0.25164844956034683617425974731834],
                [-0.00033092823209789928824847950917573,   -0.00064742428750239626181744448195926,   -0.00029378157094206654032633032158811,    0.00058803234003523551646542921316165,     0.0041848756747159772399005586544894,      0.011240277531447318158051666397921,       0.02148526549159615479120030424263,      0.026142355417612899733901322630866],
                [-0.0000021869677709226337744440535293076, -0.0000040323323856154789413751338051599, -0.0000039355521071470918751517887926294, -0.0000077356098441786850369139886516143,  -0.000006886557963029641619274086529499,   0.000020709635337561553818374320063958,    0.00013513874256402826186596446559443,    0.00022428031175139061285260626677095]]).T
    else:
        raise Exception("Can only retrieve conversion matrix for curves of order 1-7")
    return conversion_matrix


def bezier_to_minvo_control_points(bezier_control_points):
    number_of_control_points = count_number_of_control_points(bezier_control_points)
    order = number_of_control_points - 1
    conversion_matrix = get_bezier_to_minvo_conversion_matrix(order)
    bezier_control_points = np.transpose(np.dot(conversion_matrix, np.transpose(bezier_control_points)))
    return bezier_control_points

def get_bezier_to_minvo_conversion_matrix(order):
    conversion_matrix = np.array([])
    if order == 1:
        conversion_matrix = np.array([[1,0],
                                      [0,1]])
    elif order == 2:
        conversion_matrix = np.array([[   1.0773502688943426945863374365995, 0.16666666649618489281974529941922, -0.077350268894342639075186205341687],
                                [0,  0.6666666670076302698716606324194,                                    0],
                                [-0.077350268894342639075186205341687, 0.16666666649618489281974529941922,    1.0773502688943426945863374365995]]).T
    elif order == 3:
        conversion_matrix = np.array([[   1.1023313978813267298161918734325,   0.34205725283878557367983661432136, -0.092730940951574913399441894592647, -0.032032768110288671642687965618707],
        [-0.013052101283271799565459034696491,   0.61129872390869463790729840459337,   0.13937496420409472685737706009307, -0.057246528487766233562974688702143],
        [-0.057246528487766233562974688702143,   0.13937496420409472685737706009307,   0.61129872390869463790729840459337, -0.013052101283271799565459034696491],
        [-0.032032768110288671642687965618707, -0.092730940951574913399441894592647,   0.34205725283878557367983661432136,    1.1023313978813267298161918734325]]).T
    elif order == 4:
        conversion_matrix = np.array([[1.117077078336717376412908649906,   0.45648118758542358049785962914871, -0.048669493162441975885202190101899, -0.046427866222615963899907399124851, -0.025260851391442560451118534646744],
        [-0.023167130174861101744374112393051,   0.55574470086895304321035445495151,    0.2768812876322077529161122345252, -0.060949219949165079727933449885024, -0.023167130174861101744374112393051],
        [-0.045481966595552630320518651476754,  0.095151197717404381848309984029895,   0.54357641106046838963513092849348,  0.095151197717404381848309984029895, -0.045481966595552630320518651476754],
        [-0.023167130174861101744374112393051, -0.060949219949165079727933449885024,    0.2768812876322077529161122345252,   0.55574470086895304321035445495151, -0.023167130174861101744374112393051],
        [-0.025260851391442560451118534646744, -0.046427866222615963899907399124851, -0.048669493162441975885202190101899,   0.45648118758542358049785962914871,     1.117077078336717376412908649906]]).T
    elif order == 5:
        conversion_matrix = np.array([[   1.1239600590978700611596232650175,   0.55106609785799090245157930177056, -0.0028928431080551215449116599555823,  -0.043764479466951352147235514055671, -0.036716391151217086772923634647328, -0.016782730372327716350736056163878],
        [-0.029681704296433609524049818669027,   0.50519484110866368528303631268194,    0.36376322411710390917364376910644,  -0.015696998340065519913289748725562, -0.034553991899283086848781233351723, -0.017465010293905208796188817561478],
        [-0.039954179638505909123227506085295,  0.062675945214295011623741727633328,    0.49016162524231864910993265079788,    0.20842947155564948033025679705241, -0.047666501130449541097370725860486, -0.020076434496697844698375446474144],
        [-0.020076434496697844698375446474144, -0.047666501130449541097370725860486,    0.20842947155564948033025679705241,    0.49016162524231864910993265079788,  0.062675945214295011623741727633328, -0.039954179638505909123227506085295],
        [-0.017465010293905208796188817561478, -0.034553991899283086848781233351723,  -0.015696998340065519913289748725562,    0.36376322411710390917364376910644,   0.50519484110866368528303631268194, -0.029681704296433609524049818669027],
        [-0.016782730372327716350736056163878, -0.036716391151217086772923634647328,  -0.043764479466951352147235514055671, -0.0028928431080551215449116599555823,   0.55106609785799090245157930177056,    1.1239600590978700611596232650175]]).T
    elif order == 6:
        conversion_matrix = np.array([[   1.1284569548349264045857971608519,   0.62183490172906650539712598597023,  0.052220886847339666646536840593694, -0.043795656142563717468095115259864, -0.035004509179092245690227256058267, -0.022753790797390320542115890129532, -0.014282189352673140623837985252927],
        [-0.034107819214594203934906462447448,   0.45908882280702234786900961733578,   0.42663096037998418148555704236832,  0.035356951398457221257380861876505, -0.032176429881538803929650601152058, -0.025349070877075967672866146706178, -0.011717886962945370207646197048637],
        [-0.036068508607806480605875597696896,  0.039513827402238120127489760444504,   0.44157134568125875949252947505246,   0.28395027148678216518050679035642, -0.010557206583927868501255951279643,  -0.03003042535478527482335210018302, -0.014004505566174075818696784523353],
        [-0.018276045130734893137744263969304, -0.042304264909076170663672867302834,   0.15731495273597645780568826802409,   0.44897686651464880966423412374403,   0.15731495273597645780568826802409, -0.042304264909076170663672867302834, -0.018276045130734893137744263969304],
        [-0.014004505566174075818696784523353,  -0.03003042535478527482335210018302, -0.010557206583927868501255951279643,   0.28395027148678216518050679035642,   0.44157134568125875949252947505246,  0.039513827402238120127489760444504, -0.036068508607806480605875597696896],
        [-0.011717886962945370207646197048637, -0.025349070877075967672866146706178, -0.032176429881538803929650601152058,  0.035356951398457221257380861876505,   0.42663096037998418148555704236832,   0.45908882280702234786900961733578, -0.034107819214594203934906462447448],
        [-0.014282189352673140623837985252927, -0.022753790797390320542115890129532, -0.035004509179092245690227256058267, -0.043795656142563717468095115259864,  0.052220886847339666646536840593694,   0.62183490172906650539712598597023,    1.1284569548349264045857971608519]]).T
    elif order == 7:
        conversion_matrix = np.array([[    1.1303727712270086887771355845256,   0.68109926252270243980446090659594,   0.10437656210131023124460657312235, -0.034708252133669393761141396108675, -0.038987473614660572586046502804136, -0.019835182620021343050765015514852, -0.020322955223502013864530674378006,   -0.01102231756545007422319802978771],
        [ -0.037183806899682908749154014956371,   0.41860429455256576140674038589603,   0.46642971457420611466432062714163,  0.093522013122631045245741550915382, -0.028361022422121728355308985266598, -0.024539769264515698872886877203877,  -0.01632215863354738073817839877478,  -0.009703119057912593914189474974342],
        [ -0.033368155348938958214957860135466,  0.022907076769251160363753099806692,    0.3955209865447357971469621968794,   0.33905763104489074348964250216983,   0.03320738416450318204509044341513, -0.027983642523532013825805768036367, -0.020276397434973775253262921886611, -0.0098110196474418509275150457290809],
        [ -0.016971146724321654474831478105052, -0.038786111301182179327719915055042,   0.11877958073714542117664905863094,   0.41177910580217989484970084676846,   0.22449061403624685457010662187875, -0.012748249549328377839790457798792, -0.026903011251313968119975187433187,  -0.012313205983260754276269348703907],
        [ -0.012313205983260754276269348703907, -0.026903011251313968119975187433187, -0.012748249549328377839790457798792,   0.22449061403624685457010662187875,   0.41177910580217989484970084676846,   0.11877958073714542117664905863094, -0.038786111301182179327719915055042,  -0.016971146724321654474831478105052],
        [-0.0098110196474418509275150457290809, -0.020276397434973775253262921886611, -0.027983642523532013825805768036367,   0.03320738416450318204509044341513,   0.33905763104489074348964250216983,    0.3955209865447357971469621968794,  0.022907076769251160363753099806692,  -0.033368155348938958214957860135466],
        [ -0.009703119057912593914189474974342,  -0.01632215863354738073817839877478, -0.024539769264515698872886877203877, -0.028361022422121728355308985266598,  0.093522013122631045245741550915382,   0.46642971457420611466432062714163,   0.41860429455256576140674038589603,  -0.037183806899682908749154014956371],
        [  -0.01102231756545007422319802978771, -0.020322955223502013864530674378006, -0.019835182620021343050765015514852, -0.038987473614660572586046502804136, -0.034708252133669393761141396108675,   0.10437656210131023124460657312235,   0.68109926252270243980446090659594,     1.1303727712270086887771355845256]]).T
    else:
        raise Exception("Can only retrieve conversion matrix for curves of order 1-7")
    return conversion_matrix