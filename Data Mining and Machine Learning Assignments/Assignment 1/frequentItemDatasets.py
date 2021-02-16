import math
import time
from copy import copy

# INITIAL PARAMETERS
dataBase = "enron" #kos, nips, enron
K = 10
FList = [0.15, 0.1]

#GLOBAL VARIABLES
source_file = "docword." + dataBase + ".txt"
vocabulary_file = "vocab." + dataBase + ".txt"
outputfile = "output-" + dataBase
outputString = ""

itemOccurence = 0
documentArray = []
wordCount = 0
frequentItemSets = [None] * K
frequentItemWordSets = [None] * K

#########################################

def parseText(F):
	#ADD GLOBAL VARIABLES
	global documentArray
	global itemOccurence
	global wordCount
	global frequentItemSets
	global outputString

	dataSet = open(source_file, "r")
	documentCount = int(dataSet.readline())
	itemOccurence = math.ceil(F * documentCount)

	wordCount = int(dataSet.readline())
	entryCount = int(dataSet.readline())

	documentArray = [set() for i in range(documentCount)]

	for x in range(entryCount):
		entry = dataSet.readline()
		entryTuple = list(map(int, entry.split(' ')))
		documentArray[entryTuple[0]-1].add(entryTuple[1])

	#print("DOCUMENT ARRAY: " + str(documentArray))

#########################################

def doStuff():
	#ADD GLOBAL VARIABLES
	global documentArray
	global itemOccurence
	global wordCount
	global frequentItemSets
	global outputString

	freqList = [0] * wordCount

	for document in documentArray:
		for word in document:
			freqList[word-1] += 1

	tempFreqList = []

	for x in range(wordCount):
		if freqList[x] >= itemOccurence:
			tempFreqList.append({x+1})

	print(1, flush=True)
	outputString += "1\n"
	frequentItemSets[0] = tempFreqList
	print("Number of frequent sets: " + str(len(tempFreqList)), flush=True)
	outputString += ("Number of frequent sets: " + str(len(tempFreqList)) + "\n")

	for i in range(1, K):
		constructFrequentSet(i)
		if frequentItemSets[i] == []:
			break


##########################################

def constructFrequentSet(setSize):
	#ADD GLOBAL VARIABLES
	global documentArray
	global itemOccurence
	global wordCount
	global frequentItemSets
	global outputString

	print(setSize+1, flush=True)
	outputString += str(setSize+1) + "\n"
	candidateSet = constructCandidateSet(setSize)
	print("Number of candidates: " + str(len(candidateSet)), flush=True)
	outputString += ("Number of candidates: " + str(len(candidateSet))) + "\n"

	tempFreqList = []

	count = [0] * len(candidateSet)

	for document in documentArray:
		for i in range(len(candidateSet)):
			if (candidateSet[i]).issubset(document):
				count[i] += 1

	for i in range(len(candidateSet)):
		if count[i] >= itemOccurence:
			tempFreqList.append(candidateSet[i])

	print("Number of frequent sets: " + str(len(tempFreqList)), flush=True)
	outputString += ("Number of frequent sets: " + str(len(tempFreqList))) + "\n"

	frequentItemSets[setSize] = tempFreqList

#########################################

def constructCandidateSet(setSize):
	#ADD GLOBAL VARIABLES
	global documentArray
	global itemOccurence
	global wordCount
	global frequentItemSets
	global outputString

	if setSize == 1:
		tempCandidateSet = []
		for i in range(len(frequentItemSets[0])):
			for j in range(i+1, len(frequentItemSets[0])):
				tempCandidateSet.append(frequentItemSets[0][i] | frequentItemSets[0][j])
		return tempCandidateSet


	else:
		tempCandidateSet = []
		lastFrequencySet = frequentItemSets[setSize-1]

		for i in range(len(lastFrequencySet)):
			currentSet = lastFrequencySet[i]
			maxElt = max(currentSet)
			currentSmallSet = copy(currentSet)
			currentSmallSet.discard(maxElt)

			j = i+1

			while(j < len(lastFrequencySet) and currentSmallSet.issubset(lastFrequencySet[j])):
				tempLastFrequencySet = copy(currentSet)
				tempLastFrequencySet.add(max(lastFrequencySet[j]))
				tempCandidateSet.append(tempLastFrequencySet)
				j += 1

		for candidate in tempCandidateSet:
			for elt in list(candidate):
				smallerCandidate = copy(candidate)
				smallerCandidate.discard(elt)

				if smallerCandidate not in lastFrequencySet:
					tempCandidateSet.remove(candidate)

				break

		return tempCandidateSet

#########################################

def convertToWords():
	global frequentItemSets
	global frequentItemWordSets
	global vocabulary_file

	vocabularyFile = open(vocabulary_file, "r")
	vocabulary = (vocabularyFile.read()).split('\n')

	for i in range(len(frequentItemSets)):
		if frequentItemSets[i] != None:
			tempFrequentWordSet = []

			for frequentSet in frequentItemSets[i]:
				tempWordSet = set()

				for wordIndex in list(frequentSet):
					tempWordSet.add(vocabulary[wordIndex-1])

				tempFrequentWordSet.append(tempWordSet)

			frequentItemWordSets[i] = tempFrequentWordSet

def main():
	global FList
	global K
	global outputString
	global frequentItemSets
	global frequentItemWordSets

	for F in FList:
		print("F = " + str(F), flush=True)
		outputString += ("F = " + str(F)) + "\n"
		print("K = " + str(K), flush=True)
		outputString += ("K = " + str(K)) + "\n"

		startTime = time.time()
		parseText(F)
		doStuff()
		runningTime = time.time() - startTime

		print("Running Time: " + str(runningTime) + " seconds", flush=True)
		outputString += ("\n" + "Running Time: " + str(runningTime) + " seconds") + "\n"

		outputString += "\n" + str(frequentItemSets) + "\n"

		convertToWords()

		outputString += "\n" + str(frequentItemWordSets) + "\n"


		

		outputString += "\n\n"
		output = open(outputfile, "a+")
		output.write(outputString)
		print("", flush=True)
		print("", flush=True)
		#print(frequentItemSets)

if __name__ == '__main__':
  main() 