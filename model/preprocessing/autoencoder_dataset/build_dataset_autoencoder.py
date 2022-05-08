import pandas as pd
import numpy as np
from model.preprocessing.autoencoder_dataset.course_dataloader import CourseDataloader


class StudyPathAnalyzerFile:
    """
    Interface class to analyse database dumps from SIDDATA
    """
    def __init__(self):
        """
        loads database relationships (depricated for current database)
        """
        self.data_path = './src/data/db_dump_old/'
        self.course_frame = pd.read_csv(self.data_path + 'Course-2021-04-29.csv')
        self.course_membership = pd.read_csv(self.data_path + 'course_membership.csv',
                                        names=['id', 'share_social', 'share_brain', 'course_id', 'user_id'])
        self.users = pd.read_csv(self.data_path + 'SiddataUser-2021-04-29.csv')
        self.institute_membership = pd.read_csv(self.data_path + 'InstituteMembership-2021-05-02.csv')
        self.institute_id = pd.read_csv(self.data_path + 'Institute-2021-05-02.csv')
        self.field_of_study_membership = pd.read_csv(self.data_path + 'field_of_study_membership.csv')
        self.field_of_study = pd.read_csv(self.data_path +'Degree-2021-05-04.csv')

    """
    subroutine titles are self-explanatory unless description available
    """
    def get_unique_institutes(self):
        return self.institute_id['name'].unique()

    def get_institute_occurences(self):
        """
        aggregates institutes visited and their number of memberships
        :return:
        """
        occurence_frame = pd.DataFrame(columns=['id','name','count'])
        counts = self.institute_membership['institute'].value_counts()
        occurence_frame[['id','name']] = self.institute_id.loc[self.institute_id['id'].isin(counts.index)][
            ['id', 'name']]
        countlist = []
        for id in occurence_frame['id']:
            countlist.append(counts[counts.index == id][0])
        occurence_frame['count'] = countlist
        return occurence_frame

    def get_field_of_study_occurences(self):
        occurence_frame = pd.DataFrame(columns=['id','name','count'])
        counts = self.field_of_study_membership

    def get_user_courses(self, user_id, index_mode = False):
        if not index_mode:
            courses = self.course_membership.loc[self.course_membership['user_id'] == user_id]['course_id'].values
        else:
            courses = self.course_membership.loc[self.course_membership['user_id'] == user_id].index
        return courses

    def get_course_participants(self, course_id):
        members = self.course_membership.loc[self.course_membership['course_id'] == course_id]['user_id'].values
        return members

    def get_user_home_institutes(self, user_id):
        institutes = self.institute_membership.loc[self.institute_membership['user'] == user_id]['institute'].values
        names = [self.institute_id.loc[self.institute_id['id'] == index]['name'].values for index in institutes]
        return institutes, names

    def get_course_name(self, course_id):
        course_name = self.course_frame.loc[self.course_frame['id'] == course_id]['title'].values
        return course_name

    def assoc_user_institute(self, user_id):
        user_frame = pd.DataFrame(columns=['id','institute_name'])
        ids, names = self.get_user_home_institutes(user_id)
        user_frame['id'] = ids
        user_frame['institute_name'] = names
        return user_frame

    def get_institute_members(self, institute_id):
        user_ids = self.institute_membership.loc[self.institute_membership['institute'] == institute_id]['user'].values
        return user_ids

    def get_courses_per_user(self, user_list):
        new_user_courses = pd.DataFrame()
        limiter = max([len(self.get_user_courses(index)) for index in user_list])
        for index in user_list:
            user_courses = self.get_user_courses(index)
            courselist = np.append(user_courses, np.empty(limiter - len(user_courses), dtype='str'))
            for item in range(len(user_courses)):
                courselist[item] = user_courses[item]
            new_user_courses[index] = courselist
        return new_user_courses

    def get_coccurence_per_course(self, course):
        coocurences = {}
        visiting_students = self.get_course_participants(course)
        if len(visiting_students) == 0:
            return coocurences
        for student in visiting_students:
            affiliated_courses = self.get_user_courses(student,index_mode=True)
            for key in affiliated_courses:
                if not key in coocurences:
                    coocurences[key] = 1
                else: coocurences[key] += 1
        return coocurences

    def create_course_coocurence_matrix(self):
        """
        Produces a matrix representation of how often a course is visited by students who also visit another course
        :return: numpy array containing co-occurences per row
        """
        matrix = np.zeros((len(self.course_frame),len(self.course_frame)),dtype='int32')
        for index in range(len(self.course_frame)):
            current_course_id = self.course_frame.iloc[index]['id']
            increase_dict = self.get_coccurence_per_course(current_course_id)
            for key in increase_dict:
                matrix[index][key] = increase_dict[key]
        return matrix

    def create_course_dataset_per_student(self):
        return_list = []
        for student_id in self.users['id'].to_list():
            ret_frame = pd.DataFrame(columns=['user_id','field_of_study'])
            courses_visited = self.get_user_courses(student_id)
            courses = self.course_frame.loc[self.course_frame['id'].isin(courses_visited)][['id','title','ddc_code','start_time']]
            if len(courses) == 0:
                continue
            courses = courses.sort_values(by=['start_time'])
            maxlength = len(courses)
            courses = courses.reset_index(drop=True)
            f_id_r, f_nm = self.get_user_home_institutes(student_id)
            field_of_study = []
            for index in f_nm:
                if index == 'Stud.IP':
                    continue
                else:
                    field_of_study.append(index)
            if len(field_of_study) > maxlength:
                maxlength = len(field_of_study)
            field_of_study += [float('Nan')] * (maxlength-len(field_of_study))
            student_id_list = [student_id]
            student_id_list += [float('Nan')] * (maxlength -len([student_id]))
            ret_frame['user_id'] = student_id_list
            ret_frame['field_of_study'] = field_of_study
            ret_frame = pd.concat([ret_frame,courses],ignore_index=True,axis=1)
            ret_frame.columns = [['user_id','field_of_study','course_id','title','ddc_code','start_time']]
            return_list.append(ret_frame)
        return return_list

class KAnonExtractor:
    def __init__(self):
        self.data_path = './src/data/k_annonymized/'
        self.courses = pd.read_csv(self.data_path+'courses.csv',names=['id','course_number','title','description','lecturer_id','participants','type','semtree_id','semester'])
        self.course_users = pd.read_csv(self.data_path+'course_user.csv',names=['user','course'])


    def combine_data_with_explorer(self):
        pass

class NewDataExtractor:
    """
    Reads and associates database entries from database dumps
    These include:
        Courses
        Users who have visited courses
        Which courses were visited by which student

    """
    def __init__(self,text_only_mode = False, pooler_only = True, batch_size = 16):
        self.text_only_mode = text_only_mode
        self.pooler_only = pooler_only
        self.courses = pd.read_csv('./src/data/db_dump_new/course_dump.csv')
        self.institutes = pd.read_csv('./src/data/db_dump_new/institute_dump.csv')
        self.eduresource = pd.read_csv('./src/data/db_dump_new/eduresource_dump.csv')
        self.ddc_classes = pd.read_csv('./src/data/classes.tsv',dtype='str',names=['code'])
        self.ddc_lookup = self._build_ddc_class_lookup()
        self.full_frame = self._prepare_data_for_dataloader()
        self.batch_size = batch_size

    def _build_ddc_class_lookup(self):
        return_dict = {}
        for index, item in enumerate(self.ddc_classes['code'].values):
            return_dict[item] = index
        return return_dict

    def _convert_into_one_hot(self,label_list):
        """
        Converts a given DDC label into the corresponding one-hot encoding
        :param label_list: list of DDC labels to be encoded
        :return: list of one hot encodings corresponding to DDC code values
        """
        ret_list = []
        for label in label_list:
            one_hot_label = np.zeros(len(self.ddc_lookup),dtype='int')
            one_hot_label[self.ddc_lookup[label]] = 1
            ret_list.append(one_hot_label)
        return ret_list

    def _filter_courses(self):
        """
        filters courses and checks if a valid EducationalResource type object exists in the database dump
        (EducationalResource type objects are database tables that represent a variety of educational resources such as
        StudIP courses, MOOCs or OERs)
        :return: filtered dataframe containing title and DDC code
        """
        existing_courses = self.eduresource.loc[self.eduresource['id'].isin(self.courses['inheritingcourse_ptr_id'].values)]
        courses_one = existing_courses[['title','ddc_code']]
        courses_two = pd.read_csv('src/data/db_dump_new/course_dump_new.csv')
        courses_two = courses_two[['title','ddc_code']]
        filtered_frame = pd.concat([courses_one, courses_two])

        if self.text_only_mode:
            filtered_frame['ddc_code'] = self.ddc_classes['code'].iloc[0]
            return filtered_frame
        else:
            filtered_frame = filtered_frame.dropna()
            filtered_frame = filtered_frame.drop_duplicates(subset = 'title',keep='first')
            filtered_frame = filtered_frame.reset_index(drop=True)
            filtered_frame['ddc_code'] = filtered_frame['ddc_code'].str.extract('(\d+)',expand=False)
            return filtered_frame

    def _prepare_data_for_dataloader(self):
        """
        Associates all relevant information for an autoencoder network (Title, DDC code and institute the course belongs to)
        :return: dataframe containing title, ddc-one-hot encoding and institute one-hot encoding
        """
        filtered_frame = self._filter_courses()
        institutes = self._build_fake_institutes(filtered_frame['ddc_code'].to_list())
        filtered_frame['institute'] = institutes
        filtered_frame['ddc_code_one_hot'] = self._convert_into_one_hot(filtered_frame['ddc_code'].to_list())
        return filtered_frame

    def _build_fake_institutes(self, ddc_labels):
        """
        Placeholder function that adds a fake one-hot encoded institute association to a course, based on its DDC value.
        This function exists because a course-institute relationship did not exist in the database so far. Future versions will include appropriate institute association
        :param ddc_labels: list of DDC labels of courses the fake institute labels are supposed to be built for.
        :return: list of institute one hot encodings based on DDC label
        """
        out_array = []
        for label in ddc_labels:
            institute_vector = np.zeros(10,dtype='int')
            institute_vector[int(label[0])] = 1
            out_array.append(institute_vector)
        return out_array

    def get_dataloaders(self):
        """
        Seperates dataset into 90-10 for training / validation and builds Dataloader objects to feed network during training.
        See course_dataloader.py for dataloader code.
        :return: training dataloader object and validation dataloader object.
        """
        dev_indeces = np.random.choice(self.full_frame.index,len(self.full_frame)//10)
        dev_frame = self.full_frame.iloc[dev_indeces]
        train_frame = self.full_frame.loc[~self.full_frame.index.isin(dev_indeces)]

        train_tiltes = np.array(train_frame['title'].to_list(),dtype='str')
        train_insitutes = np.array(train_frame['institute'].to_list(),dtype='int32')
        train_ddc_labels = np.array(train_frame['ddc_code_one_hot'].to_list(),dtype='int32')

        dev_titles = np.array(dev_frame['title'].to_list(),dtype='str')
        dev_institutes = np.array(dev_frame['institute'].to_list(), dtype='int32')
        dev_ddc_labels = np.array(dev_frame['ddc_code_one_hot'].to_list(), dtype='int32')

        train_dataloader = CourseDataloader(titles=train_tiltes,
                                             ddc_labels=train_ddc_labels, batch_size=self.batch_size,
                                           text_only_mode=self.text_only_mode,
                                           pooler_only=self.pooler_only)

        dev_dataloader = CourseDataloader(titles=dev_titles,
                                             ddc_labels=dev_ddc_labels, batch_size=self.batch_size,
                                           text_only_mode=self.text_only_mode,
                                           pooler_only=self.pooler_only)
        return train_dataloader, dev_dataloader