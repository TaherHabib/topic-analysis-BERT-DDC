import pandas as pd
import numpy as np

class PreprocessingDatahandler:
    def __init__(self):
        self.data_path = './src/data/p3_dump/'
        self.users = pd.read_csv(self.data_path + 'SiddataUser-2021-11-15.csv')
        self.course_membership = pd.read_csv(self.data_path + 'CourseMembership-2021-11-15.csv')
        self.eduresources = pd.read_csv(self.data_path + 'EducationalResource-2021-11-15.csv')
        self.classes = pd.read_csv('./src/data/classes.tsv', dtype='str', names=['ddc_code'])
        self.ddc_lookup = self._build_ddc_class_lookup()
        self.courses = self._course_preprocess()

    def course_get_associated_ddc(self,course_ptr_ids):
        ddc = []
        for index in course_ptr_ids:
            ddc_id = self.eduresources.loc[self.eduresources['id'] == index]['ddc_code'].values[0]
            ddc_id.replace('"', '')
            ddc.append(ddc_id)
        return ddc

    def _course_preprocess(self):
        c = pd.read_csv(self.data_path + 'StudipCourse-2021-11-15.csv')
        courses = c[['id','educationalresource_ptr','title','description']]
        courses['ddc'] = self.course_get_associated_ddc(courses['educationalresource_ptr'].to_list())
        return courses

    def _build_ddc_class_lookup(self):
        return_dict = {}
        for index, item in enumerate(self.classes['ddc_code'].values):
            return_dict[item] = index
        return return_dict

    def convert_into_one_hot(self, label_list):
        """
        Converts a given DDC label into the corresponding one-hot encoding
        :param label_list: list of DDC labels to be encoded
        :return: list of one hot encodings corresponding to DDC code values
        """
        ret_list = []
        for label in label_list:
            one_hot_label = np.zeros(len(self.ddc_lookup), dtype='int')
            one_hot_label[self.ddc_lookup[label]] = 1
            ret_list.append(one_hot_label)
        return ret_list

    def eduresource_get_courses_only(self):
        return self.eduresources.loc[self.eduresources['type'] == "['SIP']"]

    def course_append_old_db_dumb_and_clean(self):
        file_path = './src/data/db_dump_new/'
        old_courses = pd.read_csv(file_path+'course_dump_new.csv')
        old_courses = old_courses[['id','title','description', 'ddc_code']]
        old_courses['educationalresource_ptr'] = ''
        old_courses['ddc_code'] = old_courses['ddc_code'].str.replace('\\', '')
        old_courses.rename(columns={'ddc_code': 'ddc'}, inplace=True)
        concat_courses = pd.concat([old_courses, self.courses]).reset_index(drop=True)
        concat_courses = concat_courses.drop_duplicates(subset='title',ignore_index=True)
        return concat_courses


    def course_generate_dataset(self, course_frame, dev_percent = 0.1):
        if dev_percent < 0 or dev_percent > 1:
            raise ValueError("value for validation set percentage out of bounds, must be between 0 and 1.")
        course_frame = course_frame.dropna(subset=['ddc','title'])
        dev_indeces = np.random.choice(len(course_frame), round(len(course_frame) * dev_percent))
        dev_frame = course_frame.iloc[dev_indeces]
        train_frame = course_frame.loc[~course_frame.index.isin(dev_indeces)]
        train_titles = np.array(train_frame['title'].to_list(),dtype='str')
        train_ddc = np.array(self.convert_into_one_hot(train_frame['ddc'].str.replace('"', '')),dtype='int32')
        dev_titles = np.array(dev_frame['title'].to_list(),dtype='str')
        dev_ddc = np.array(self.convert_into_one_hot(dev_frame['ddc'].str.replace('"', '')),dtype='int32')
        return {'train': {'title': train_titles, 'ddc': train_ddc}, 'dev': {'title': dev_titles, 'ddc': dev_ddc}}

    def course_get_dataset(self, include_old=True):
        if include_old:
            set = self.course_append_old_db_dumb_and_clean()
        else:
            set = self.courses
        return self.course_generate_dataset(set, dev_percent=0.1)

    def studypath_collect_user_courses(self):
        pass


#integrate old dataset
#train new autoencoder?