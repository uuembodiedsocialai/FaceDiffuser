import os
import shutil
import numpy as np

path_to_save = 'vertices_npy'

mapping = {'SEN_alfalfa_is_healthy_for_you': '01',
           'SEN_allow_each_child_to_have_an_ice_pop': '02',
           'SEN_all_your_wishful_thinking_wont_change_that': '03',
           'SEN_and_you_think_you_have_language_problems': '04',
           'SEN_approach_your_interview_with_statuesque_composure': '05',
           'SEN_are_you_looking_for_employment': '06',
           'SEN_as_she_drove_she_thought_about_her_plan': '07',
           'SEN_a_good_morrow_to_you_my_boy': '08',
           'SEN_a_voice_spoke_near-at-hand': '09',
           'SEN_both_figures_would_go_higher_in_later_years': '10',
           'SEN_boy_youre_stirrin_early_a_sleepy_voice_said': '11',
           'SEN_by_eating_yogurt_you_may_live_longer': '12',
           'SEN_cliff_was_soothed_by_the_luxurious_massage': '13',
           'SEN_did_Shawn_catch_that_big_goose_without_help': '14',
           'SEN_do_they_make_class_biased_decisions': '15',
           'SEN_drop_five_forms_in_the_box_before_you_go_out': '16',
           'SEN_george_is_paranoid_about_a_future_gas_shortage': '17',
           'SEN_go_change_your_shoes_before_you_turn_around': '18',
           'SEN_greg_buys_fresh_milk_each_weekday_morning': '19',
           'SEN_have_you_got_our_keys_handy': '20',
           'SEN_however_a_boys_lively_eyes_might_rove': '21',
           'SEN_how_do_oysters_make_pearls': '22',
           'SEN_how_long_would_it_be_occupied': '23',
           'SEN_how_ya_gonna_keep_em_down_on_the_farm': '24',
           'SEN_id_rather_not_buy_these_shoes_than_be_overcharged': '25',
           'SEN_if_dark_came_they_would_lose_her': '26',
           'SEN_im_going_to_search_this_house': '27',
           'SEN_its_healthier_to_cook_without_sugar': '28',
           'SEN_jeffs_toy_go_cart_never_worked': '29',
           'SEN_more_he_could_take_at_leisure': '30',
           'SEN_nobody_else_showed_pleasure': '31',
           'SEN_oh_we_managed_she_said': '32',
           'SEN_she_always_jokes_about_too_much_garlic_in_his_food': '33',
           'SEN_take_charge_of_choosing_her_bridesmaids_gowns': '34',
           'SEN_thank_you_she_said_dusting_herself_off': '35',
           'SEN_then_he_thought_me_more_perverse_than_ever': '36',
           'SEN_theyre_going_to_louse_me_up_good': '37',
           'SEN_theyve_never_met_you_know': '38',
           'SEN_they_all_like_long_hot_showers': '39',
           'SEN_they_are_both_trend_following_methods': '40',
           'SEN_they_enjoy_it_when_I_audition': '41',
           'SEN_they_had_slapped_their_thighs': '42',
           'SEN_they_werent_as_well_paid_as_they_should_have_been': '43',
           'SEN_the_small_boy_put_the_worm_on_the_hook': '44',
           'SEN_when_she_awoke_she_was_the_ship': '45',
           'SEN_why_buy_oil_when_you_always_use_mine': '46',
           'SEN_why_charge_money_for_such_garbage': '47',
           'SEN_why_put_such_a_high_value_on_being_top_dog': '48',
           'SEN_with_each_song_he_gave_verbal_footnotes': '49',
           'SEN_youre_boiling_milk_aint_you': '50'}


# statistics over the values of the coordinates
maxs = [125.922, 185.79601, 75.365097]
mins = [-117.798, -213.735, -177.091]


def valid(a):
    return a.strip() != ""


def spl(a, fn):
    return list(map(fn, filter(valid, a.split(" "))))


def create_sequence(sequence_path, face_id):
    sequence = []
    nr_values = 0
    for file in os.listdir(sequence_path):
        if file.endswith('.obj'):
            vertices = []
            mesh_file = open(os.path.join(sequence_path, file))

            for line in mesh_file:
                if line.startswith('v '):
                    coords = spl(line[1:], float)

                    for idx in range(3):
                        X_std = (coords[idx] - mins[idx]) / (maxs[idx] - mins[idx])
                        X_scaled = X_std * 2 - 1
                        coords[idx] = X_scaled

                    vertices += coords
            nr_values += 1
            sequence.append(vertices)
    seq = np.array(sequence)
    seq_filename = os.path.join(path_to_save, f"{face_id}_{mapping[os.path.basename(sequence_path)]}.npy")
    np.save(seq_filename, seq)
    print(f"Done {seq_filename}")
    return nr_values


def create_subject_sequences(subject):
    audio_path = os.path.join(subject, 'tracked_mesh')
    index = 1
    for x in os.scandir(audio_path):
        shutil.copy(x,
                    os.path.join(path_to_save, f"{os.path.basename(subject)}_{mapping[os.path.basename(x)[:-4]]}.wav")
                    )
        create_sequence(x.path, f"{index:02d}")
        index += 1


if __name__ == '__main__':
    path_to_dataset = 'multiface' # path to downloaded set
    subjects = [x.path for x in os.scandir(path_to_dataset)]
    for s in subjects:
        create_subject_sequences(s)
