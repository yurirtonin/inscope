import numpy as np

import misc
import cdi

def call_pie(DPs, positions, initial_obj, initial_probe,iterations):

    misc.check_data(DPs, positions, initial_obj, initial_probe)

    obj = np.empty(initial_probe.shape[0],initial_obj.shape[0],initial_obj.shape[1]) # n_modes, rows, columns
    obj[:] = initial_obj

    probe = initial_probe
    size_y, size_x = probe[0].shape

    error = [] 
    for iter in range(iterations):

        print(f'Iteration #{iter+1}/{iterations}...')

        for idx, pos_x, pos_y in enumerate(zip(positions)):

            obj_roi = obj[:,pos_y:pos_y+size_y,pos_x:pos_x+size_x]

            wavefront = obj_roi*probe

            updated_wavefront = cdi.wavefront_update(DPs[idx],wavefront)

            obj_roi, probe = pie_update_obj_and_probe(obj_roi, probe, wavefront, updated_wavefront)
        
            obj[:,pos_y:pos_y+size_y,pos_x:pos_x+size_x] = obj_roi


        error.append(misc.mean_square_error())


    return obj, probe, error

def pie_update_obj_and_probe(obj,probe,wavefront,updated_wavefront):

    delta_wavefront = updated_wavefront - wavefront

    updated_obj = obj + probe.conj()*delta_wavefront/( (1-reg_) )

    return 0 