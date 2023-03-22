import argparse
import mir_eval
import mido
import numpy as np
from glob import glob
from os.path import join as joinPath
import os
from mir_eval.onset import evaluate as evaluate_onsets
from sklearn.metrics import accuracy_score
from math import ceil, isnan
from pretty_midi import PrettyMIDI

MINIMUM_MIDI = 21
SMALLEST_STEP_DURATION = 0.032

def get_times_freq_from_midi_data(midi_data):
	times = np.unique(midi_data[:,0])
	all_freqs = []
	for _time in times:
		pitches = midi_data[np.where(midi_data[:,0]==_time)][:,2].astype(int)
		freqs = midi2hz(pitches + MINIMUM_MIDI)
		all_freqs.append(freqs)
	return times, all_freqs

def midi2hz(notes):
	return 440.0 * (2.0 ** ((np.asanyarray(notes) - 69.0) / 12.0))

def calculate_f1(precision, recall):
	f1 = 2 * (precision * recall) / (precision + recall)
	if isnan(f1):
		f1=0.0
	return f1

def mean_list(_list):
	return round(sum(_list) / len(_list), 2)

def parse_midi(path):
	"""open midi file and return np.array of (onset, offset, note, velocity) rows"""
	midi = mido.MidiFile(path)
	time = 0
	sustain = False
	events = []
	for message in midi:
		time += message.time

		if message.type == 'control_change' and message.control == 64 and (message.value >= 64) != sustain:
			sustain = message.value >= 64
			event_type = 'sustain_on' if sustain else 'sustain_off'
			event = dict(index=len(events), time=time, type=event_type, note=None, velocity=0)
			events.append(event)

		if 'note' in message.type:
			velocity = message.velocity if message.type == 'note_on' else 0
			event = dict(index=len(events), time=time, type='note', note=message.note, velocity=velocity, sustain=sustain)
			events.append(event)

	notes = []
	for i, onset in enumerate(events):
		if onset['velocity'] == 0:
			continue
		offset = next(n for n in events[i + 1:] if n['note'] == onset['note'] or n is events[-1])
		note = (onset['time'], offset['time'], onset['note'], onset['velocity'])
		notes.append(note)

	return np.array(notes)

def save_evaluation_results(ScoresIgnoringPitches, Scores, save_file_path, use_only_wrt_pitches):

	headers = ', '.join(['threshold', 'F1', 'Precision', 'Recall', 'Accuracy', 'w.r.t pitches'])
	output_data = []
	#	 save_file_path
	for thr_i, thr in enumerate(ScoresIgnoringPitches):

		if not use_only_wrt_pitches:
			_scores = ScoresIgnoringPitches[thr]
			F1 = mean_list(_scores['F1'])
			Precision = mean_list(_scores['Precision'])
			Recall = mean_list(_scores['Recall'])
			Accuracy = mean_list(_scores['Accuracy'])
			output_data.append([thr, F1, Precision, Recall, Accuracy, "No"])

		_scores = Scores[thr]
		F1 = mean_list(_scores['F1'])
		Precision = mean_list(_scores['Precision'])
		Recall = mean_list(_scores['Recall'])
		Accuracy = mean_list(_scores['Accuracy'])
		output_data.append([thr, F1, Precision, Recall, Accuracy, "Yes"])

	output_data = np.array(output_data, dtype=str)

	np.savetxt(save_file_path, output_data, header=headers, fmt='%s, %s, %s, %s, %s, %s')


def main(sp_folder, tp_folder, save_file_path, use_only_wrt_pitches):

	sp_midis = glob(joinPath(sp_folder, '*mid'))
	tp_midis = glob(joinPath(tp_folder, '*mid'))

	tp_thresholds = []
	tp_midis_patterns = []

	sp_midi_name = sp_midis[0].split('/')[-1].split('.')[0]
	tp_midis_re = joinPath(tp_folder, sp_midi_name+'*.mid')
	tp_midis_for_this_sp = glob(tp_midis_re)
	sp_part_to_be_replaced = "_SP_MIDI_NAME_"

	# figure out what are the included thresholds in tp folder
	for tp_midi in tp_midis_for_this_sp:
		threshold_str = tp_midi.split('_estimated_')[-1].split('.mid')[0]
		try:
			threshold = float(threshold_str)
			tp_thresholds.append(threshold)
			tp_midis_patterns.append(joinPath(tp_folder, sp_part_to_be_replaced+'_estimated_'+threshold_str+'.mid'))
		except:
			print(f"ERROR, Threshold extracted from file {tp_midi} is {threshold_str}, which is not a float")
			print(f"SP midi:	  {sp_midis[0]}")
			print(f"SP midi name: {sp_midi_name}")
			
	# group sp, tp midis based on threshold value
	sp_tp_midis_groups = {}
	for thr_i, thr in enumerate(tp_thresholds):
		sp_tp_midis_groups[thr] = []
		for sp_midi in sp_midis:
			sp_midi_name = sp_midis[0].split('/')[-1].split('.')[0]
			tp_midi = tp_midis_patterns[thr_i].replace(sp_part_to_be_replaced, sp_midi_name)
			if os.path.isfile(tp_midi):
				sp_tp_midis_groups[thr].append([sp_midi, tp_midi])


	Scores = {}
	ScoresIgnoringPitches = {}

	for thr_i, thr in enumerate(sp_tp_midis_groups):

		ScoresIgnoringPitches[thr] = {}
		ScoresIgnoringPitches[thr]['F1'] = []
		ScoresIgnoringPitches[thr]['Precision'] = []
		ScoresIgnoringPitches[thr]['Recall'] = []
		ScoresIgnoringPitches[thr]['Accuracy'] = []

		Scores[thr] = {}
		Scores[thr]['F1'] = []
		Scores[thr]['Precision'] = []
		Scores[thr]['Recall'] = []
		Scores[thr]['Accuracy'] = []

		for midi_i, (sp_midi, tp_midi) in enumerate(sp_tp_midis_groups[thr]):

			sp_data = parse_midi(sp_midi)
			tp_data = parse_midi(tp_midi)


			# Ignoring pitches
			sp_onsets = sp_data[:,0]
			tp_onsets = tp_data[:,0]

			end_time = max(sp_onsets[-1], tp_onsets[-1])
			n_frames = ceil(end_time/SMALLEST_STEP_DURATION)

			# calculate F1, precision, recall part
			scores = evaluate_onsets(sp_onsets, tp_onsets)
			ScoresIgnoringPitches[thr]['F1'].append(scores['F-measure'])
			ScoresIgnoringPitches[thr]['Precision'].append(scores['Precision'])
			ScoresIgnoringPitches[thr]['Recall'].append(scores['Recall'])

			# calculate accuracy part
			sp_frames = np.zeros((n_frames))
			sp_indexes_on = (sp_onsets//SMALLEST_STEP_DURATION).astype(int)
			sp_frames[sp_indexes_on] = 1
			tp_frames = np.zeros((n_frames))
			tp_indexes_on = (tp_onsets//SMALLEST_STEP_DURATION).astype(int)
			tp_frames[tp_indexes_on] = 1

			ScoresIgnoringPitches[thr]['Accuracy'].append(accuracy_score(sp_frames, tp_frames))

			# Considering pitches
			ref_time, ref_freqs = get_times_freq_from_midi_data(sp_data)
			est_time, est_freqs = get_times_freq_from_midi_data(tp_data)

			scores = mir_eval.multipitch.evaluate(ref_time, ref_freqs, est_time, est_freqs,)
			f1 = calculate_f1(scores['Precision'], scores['Recall'])

			Scores[thr]['F1'].append(f1)
			Scores[thr]['Precision'].append(scores['Precision'])
			Scores[thr]['Recall'].append(scores['Recall'])
			Scores[thr]['Accuracy'].append(scores['Accuracy'])


	save_evaluation_results(ScoresIgnoringPitches, Scores, save_file_path, use_only_wrt_pitches)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-SP", "--SP_MIDI_Folder", required=True, type=str)
	parser.add_argument("-TP", "--TP_MIDI_Folder", required=True, type=str)
	parser.add_argument("-O", "--Output_File_Path", required=True, type=str)
	parser.add_argument("-OP", "--Use_only_wrt_pitches", action='store_true')
	parser.add_argument('--no-OP', dest='Use_only_wrt_pitches', action='store_false')
	parser.set_defaults(Use_only_wrt_pitches=True)
	args = parser.parse_args()

	main(args.SP_MIDI_Folder, args.TP_MIDI_Folder, args.Output_File_Path, args.Use_only_wrt_pitches)


'''
python evaluate_onset_model_midis.py -SP test_example/reference_mids  \
    -TP test_example/estimated_mids -O results.csv -OP

python evaluate_onset_model_midis.py -SP test_example/reference_mids  \
    -TP test_example/estimated_mids -O results.csv --no-OP
'''
