import React, { useState } from "react";
import { Container, Center, VStack, StackDivider, CircularProgress, CircularProgressLabel, useBoolean, Text, Box } from '@chakra-ui/react'

const regx = new RegExp(/.*base64,/);
const url = 'http://127.0.0.1:8000/'
const fileFormat = ['doc', 'docx', 'txt']
export default function App() {
  const [jsonResponse, setJsonResponse] = useState({response: [], paragraphs: '', candidate: [], topic: ''});
  const [loading, setLoading] = useBoolean()
  const [hasData, setHasData] = useBoolean()
  const showFile = (e) => {
    if (e.target.files.length === 0) return;
    setLoading.on();
    e.preventDefault();
    window.fileName = e.target.files[0].name
    const reader = new FileReader();
    reader.onload = (e) => {
      const text = e.target.result;
      let format = window.fileName.split('.')[1]
      postFile(text.replace(regx, ''), format).then(json_data => {
        console.log(json_data)
        setJsonResponse(json_data);
        setLoading.off();
        setHasData.on();
      })
    };
    
    reader.readAsDataURL(e.target.files[0]);
  };
  return (
    <Container color='gray.800' pt={10} maxW='container.lg'>
      <VStack paddingBlockEnd={100}>
        <Center>
          <Box>
            <input type="file" onChange={showFile} />
          </Box>
        </Center>
        
          <Box borderRadius='md' color='black.700'>
          {(loading) ?
            <Center>
              <CircularProgress isIndeterminate color='gray.900' size='200px' pt={5}>
                <CircularProgressLabel color='gray.600'>Loading</CircularProgressLabel>
              </CircularProgress>
            </Center>
            :
            (hasData)?
            <VStack divider={<StackDivider borderColor='gray.200' />} alignItems="baseline">
              <Box boxShadow={'xl'} rounded={'md'} overflow={'hidden'} padding={15} width="-webkit-fill-available"><Text>&emsp; {jsonResponse['paragraphs']}</Text></Box>
              <Box boxShadow={'xl'} rounded={'md'} overflow={'hidden'} padding={15} width="-webkit-fill-available">
                {jsonResponse['response'].map((text, i) => <Box key={i}><Text>Topics {i+1}: {text}</Text></Box>)}
              </Box>
              <Box boxShadow={'xl'} rounded={'md'} overflow={'hidden'} padding={15} width="-webkit-fill-available">
                {jsonResponse['candidate'].map((text, i) => <Box key={i}><Text>Candidate {i+1}: {text}</Text></Box>)}
              </Box>
              <Box boxShadow={'xl'} rounded={'md'} overflow={'hidden'} padding={15} width="-webkit-fill-available"><Text>{jsonResponse['topic']}</Text></Box>
            </VStack>
            :
            <>
            </>
          }
          </Box>
      </VStack>
    </Container>
  )
}


async function postFile(data, format) {
  if(!search(format, fileFormat)){
    return {'response': 'file format not found!!!'}
  }
  const response = await fetch(url+format, {
    method: 'POST',
    body: JSON.stringify({ file: data })
  })
  return response.json(response);
}

function search(key, list){
  for(let i in list){
    if(list[i] === key){
      return true;
    }
  }
  return false
}