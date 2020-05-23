import pandas as pd
import requests
from tqdm import tqdm
from urllib.parse import urljoin
from typing import *
from pprint import pprint



URL_LIST_COMPETITION = 'https://www.kaggle.com/requests/CompetitionService/ListCompetitions'



def get_competition_info(n=50) -> List[dict]:

	header = {
		'content-type': 'application/json',
		'origin': 'https://www.kaggle.com',
		'pragma': 'no-cache',
		'referer': 'https://www.kaggle.com/competitions',
		'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.138 Safari/537.36',
		'sec-fetch-dest': 'empty',
		'sec-fetch-mode': 'cors',
		'sec-fetch-site': 'same-origin',
		'cookie': 'CSRF-TOKEN=CfDJ8LdUzqlsSWBPr4Ce3rb9VL8yrRXOJhJolJx6sdhu3iRAPvLtPThUGpv5Pzp-2AMyBkHLeFXIQzWOoyVO7GC3V1o3ShJ4zDdQB0If6P77eL6haOFmzIRS3AkTr82p7wvud3MZndl17zfupZH58rgXbc8; GCLB=CLzHw8Tqi8DBrQE; ka_sessionid=c38d7fd5182140710d170a62f9c679ef2d8c18d1; XSRF-TOKEN=CfDJ8LdUzqlsSWBPr4Ce3rb9VL_CXdYFqryoq_P5GlfUKZ16J2ueB0OF4deOc2u32nUWhJe1LHqEYOVptRG0cwnTTeCLgtpt-AdviT4Ds08jxk7kL0FFQoaIfg4FIQDnBDLCHSWEuB84sP-ZVSVjFsyNBP8; CLIENT-TOKEN=eyJhbGciOiJub25lIiwidHlwIjoiSldUIn0.eyJpc3MiOiJrYWdnbGUiLCJhdWQiOiJjbGllbnQiLCJzdWIiOm51bGwsIm5idCI6IjIwMjAtMDUtMjNUMDI6MjY6MzUuMjgzNzE0N1oiLCJpYXQiOiIyMDIwLTA1LTIzVDAyOjI2OjM1LjI4MzcxNDdaIiwianRpIjoiZmExYjg2ZDYtZGI1NS00NTQwLTk5ZGEtNmNjOWFjNjIzYWNiIiwiZXhwIjoiMjAyMC0wNi0yM1QwMjoyNjozNS4yODM3MTQ3WiIsImFub24iOnRydWUsImZmIjpbIkZsZXhpYmxlR3B1IiwiS2VybmVsc0ludGVybmV0IiwiRGF0YUV4cGxvcmVyVjIiLCJEYXRhU291cmNlU2VsZWN0b3JWMiIsIktlcm5lbHNWaWV3ZXJJbm5lclRhYmxlT2ZDb250ZW50cyIsIkZvcnVtV2F0Y2hEZXByZWNhdGVkIiwiTmV3S2VybmVsV2VsY29tZSIsIk1kZUltYWdlVXBsb2FkZXIiLCJLZXJuZWxzUXVpY2tWZXJzaW9ucyIsIkRpc2FibGVDdXN0b21QYWNrYWdlcyIsIkRvY2tlck1vZGFsU2VsZWN0b3IiLCJQaG9uZVZlcmlmeUZvckdwdSIsIkNsb3VkU2VydmljZXNLZXJuZWxJbnRlZyIsIlVzZXJTZWNyZXRzS2VybmVsSW50ZWciLCJOYXZpZ2F0aW9uUmVkZXNpZ24iLCJLZXJuZWxzU25pcHBldHMiLCJLZXJuZWxXZWxjb21lTG9hZEZyb21VcmwiLCJUcHVLZXJuZWxJbnRlZyIsIktlcm5lbHNGaXJlYmFzZVByb3h5IiwiS2VybmVsc0ZpcmViYXNlTG9uZ1BvbGxpbmciLCJEYXRhc2V0TGl2ZU1vdW50IiwiRW5hYmxlUmFwaWRhc2giLCJEYXRhc2V0c0RhdGFFeHBsb3JlclYzVHJlZUxlZnQiXSwicGlkIjoia2FnZ2xlLTE2MTYwNyIsInN2YyI6IndlYi1mZSIsInNkYWsiOiJBSXphU3lEQU5HWEZIdFNJVmM1MU1JZEd3ZzRtUUZnbTNvTnJLb28iLCJibGQiOiIxZTBkZDUzYWMzNTQzZTg3YjI5NzIzMDQzZGI3YjAzYWY2OWI5YzM4In0.',
		'__requestverificationtoken': 'CfDJ8LdUzqlsSWBPr4Ce3rb9VL_CXdYFqryoq_P5GlfUKZ16J2ueB0OF4deOc2u32nUWhJe1LHqEYOVptRG0cwnTTeCLgtpt-AdviT4Ds08jxk7kL0FFQoaIfg4FIQDnBDLCHSWEuB84sP-ZVSVjFsyNBP8',
		'x-xsrf-token': 'CfDJ8LdUzqlsSWBPr4Ce3rb9VL_CXdYFqryoq_P5GlfUKZ16J2ueB0OF4deOc2u32nUWhJe1LHqEYOVptRG0cwnTTeCLgtpt-AdviT4Ds08jxk7kL0FFQoaIfg4FIQDnBDLCHSWEuB84sP-ZVSVjFsyNBP8',
	}


	params = {
		'pageSize': n,
		'pageToken': '001'
	}


	resp = requests.post(URL_LIST_COMPETITION, headers=header, json=params)
	competitions = resp.json()['result']['competitions']


	def parse_single_competition_info(info: dict) -> dict:
		res = {}
		
		copy_fields = ['title', 'briefDescription', 'id', 'totalCompetitors', 'hasScripts', 'hasSolution']
		for k in copy_fields:
			res[k] = info.get(k, None)

		# competition category is a list
		categories = info.get('categories', [])
		categories = [c['displayName'] for c in categories]
		res['categories'] = categories

		return res


	competition_info = [parse_single_competition_info(c) for c in competitions]
	return competition_info



def get_kernel_info(competition_id: int) -> dict:
	url = 'https://www.kaggle.com/kernels.json?sortBy=hotness&group=everyone&pageSize=100&competitionId={}'.format(competition_id)
	resp = requests.get(url)
	kernels = resp.json()

	pprint(resp.json())

	def parse_single_kernel_info(info: dict) -> dict:
		res = {}

		copy_fields = ['languageName', 'id', 'isNotebook', 'scriptCommentsUrl', 'scriptUrl', 'title', 
						'totalComments', 'totalVotes', 'totalViews']
		for k in copy_fields:
			res[k] = info.get(k, None)

		res['author'] = info['author']['userName']
		res['notebookFullUrl'] = urljoin('https://www.kaggle.com/', info['scriptUrl'])
	
		return res

	return [parse_single_kernel_info(k) for k in kernels]






print(get_kernel_info(6392)[1])


# url = 'https://www.kaggle.com/'
# client = requests.session()
# # Retrieve the CSRF token first
# pprint(client.get(url).cookies)


