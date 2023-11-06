
#ifndef BUFFER_STUFFER
#define BUFFER_STUFFER

#include "NanoLog.h"
#include "Packer.h"

#include <string>

// Since some of the functions/variables output below are for debugging purposes
// only (i.e. they're not used in their current form), squash all gcc complaints
// about unused variables/functions.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-variable"

/**
 * Describes a log message found in the user sources by the original format
 * string provided, the file where the log message occurred, and the line number
 */
struct LogMetadata {
  const char *fmtString;
  const char *fileName;
  uint32_t lineNumber;
  NanoLog::LogLevel logLevel;
};

// Start an empty namespace to enclose all the record(debug)/compress/decompress
// and support functions
namespace {

using namespace NanoLog::LogLevels;

inline void __syang0__fl__Debug32level__testHelper47client46cc__23__(NanoLog::LogLevel level, const char* fmtStr ) {
    extern const uint32_t __fmtId__Debug32level__testHelper47client46cc__23__;

    if (level > NanoLog::getLogLevel())
        return;

    uint64_t timestamp = PerfUtils::Cycles::rdtsc();
    ;
    size_t allocSize =   sizeof(NanoLogInternal::Log::UncompressedEntry);
    NanoLogInternal::Log::UncompressedEntry *re = reinterpret_cast<NanoLogInternal::Log::UncompressedEntry*>(NanoLogInternal::RuntimeLogger::reserveAlloc(allocSize));

    re->fmtId = __fmtId__Debug32level__testHelper47client46cc__23__;
    re->timestamp = timestamp;
    re->entrySize = static_cast<uint32_t>(allocSize);

    char *buffer = re->argData;

    // Record the non-string arguments
    

    // Record the strings (if any) at the end of the entry
    

    // Make the entry visible
    NanoLogInternal::RuntimeLogger::finishAlloc(allocSize);
}


inline ssize_t
compressArgs__Debug32level__testHelper47client46cc__23__(NanoLogInternal::Log::UncompressedEntry *re, char* out) {
    char *originalOutPtr = out;

    // Allocate nibbles
    BufferUtils::TwoNibbles *nib = reinterpret_cast<BufferUtils::TwoNibbles*>(out);
    out += 0;

    char *args = re->argData;

    // Read back all the primitives
    

    // Pack all the primitives
    

    if (false) {
        // memcpy all the strings without compression
        size_t stringBytes = re->entrySize - ( 0)
                                            - sizeof(NanoLogInternal::Log::UncompressedEntry);
        if (stringBytes > 0) {
            memcpy(out, args, stringBytes);
            out += stringBytes;
        }
    }

    return out - originalOutPtr;
}


inline void
decompressPrintArgs__Debug32level__testHelper47client46cc__23__ (const char **in,
                        FILE *outputFd,
                        void (*aggFn)(const char*, ...)) {
    BufferUtils::TwoNibbles nib[0];
    memcpy(&nib, (*in), 0);
    (*in) += 0;

    // Unpack all the non-string argments
    

    // Find all the strings
    

    const char *fmtString = "Debug level";
    const char *filename = "testHelper/client.cc";
    const int linenum = 23;
    const NanoLog::LogLevel logLevel = DEBUG;

    if (outputFd)
        fprintf(outputFd, "Debug level" "\r\n" );

    if (aggFn)
        (*aggFn)("Debug level" );
}


inline void __syang0__fl__Error32Level__testHelper47client46cc__26__(NanoLog::LogLevel level, const char* fmtStr ) {
    extern const uint32_t __fmtId__Error32Level__testHelper47client46cc__26__;

    if (level > NanoLog::getLogLevel())
        return;

    uint64_t timestamp = PerfUtils::Cycles::rdtsc();
    ;
    size_t allocSize =   sizeof(NanoLogInternal::Log::UncompressedEntry);
    NanoLogInternal::Log::UncompressedEntry *re = reinterpret_cast<NanoLogInternal::Log::UncompressedEntry*>(NanoLogInternal::RuntimeLogger::reserveAlloc(allocSize));

    re->fmtId = __fmtId__Error32Level__testHelper47client46cc__26__;
    re->timestamp = timestamp;
    re->entrySize = static_cast<uint32_t>(allocSize);

    char *buffer = re->argData;

    // Record the non-string arguments
    

    // Record the strings (if any) at the end of the entry
    

    // Make the entry visible
    NanoLogInternal::RuntimeLogger::finishAlloc(allocSize);
}


inline ssize_t
compressArgs__Error32Level__testHelper47client46cc__26__(NanoLogInternal::Log::UncompressedEntry *re, char* out) {
    char *originalOutPtr = out;

    // Allocate nibbles
    BufferUtils::TwoNibbles *nib = reinterpret_cast<BufferUtils::TwoNibbles*>(out);
    out += 0;

    char *args = re->argData;

    // Read back all the primitives
    

    // Pack all the primitives
    

    if (false) {
        // memcpy all the strings without compression
        size_t stringBytes = re->entrySize - ( 0)
                                            - sizeof(NanoLogInternal::Log::UncompressedEntry);
        if (stringBytes > 0) {
            memcpy(out, args, stringBytes);
            out += stringBytes;
        }
    }

    return out - originalOutPtr;
}


inline void
decompressPrintArgs__Error32Level__testHelper47client46cc__26__ (const char **in,
                        FILE *outputFd,
                        void (*aggFn)(const char*, ...)) {
    BufferUtils::TwoNibbles nib[0];
    memcpy(&nib, (*in), 0);
    (*in) += 0;

    // Unpack all the non-string argments
    

    // Find all the strings
    

    const char *fmtString = "Error Level";
    const char *filename = "testHelper/client.cc";
    const int linenum = 26;
    const NanoLog::LogLevel logLevel = ERROR;

    if (outputFd)
        fprintf(outputFd, "Error Level" "\r\n" );

    if (aggFn)
        (*aggFn)("Error Level" );
}


inline void __syang0__fl__I32have32a32couple32of32things3237d443237f443237u443237s__testHelper47client46cc__31__(NanoLog::LogLevel level, const char* fmtStr , int arg0, double arg1, unsigned int arg2, const char* arg3) {
    extern const uint32_t __fmtId__I32have32a32couple32of32things3237d443237f443237u443237s__testHelper47client46cc__31__;

    if (level > NanoLog::getLogLevel())
        return;

    uint64_t timestamp = PerfUtils::Cycles::rdtsc();
    size_t str3Len = 1 + strlen(arg3);;
    size_t allocSize = sizeof(arg0) + sizeof(arg1) + sizeof(arg2) +  str3Len +  sizeof(NanoLogInternal::Log::UncompressedEntry);
    NanoLogInternal::Log::UncompressedEntry *re = reinterpret_cast<NanoLogInternal::Log::UncompressedEntry*>(NanoLogInternal::RuntimeLogger::reserveAlloc(allocSize));

    re->fmtId = __fmtId__I32have32a32couple32of32things3237d443237f443237u443237s__testHelper47client46cc__31__;
    re->timestamp = timestamp;
    re->entrySize = static_cast<uint32_t>(allocSize);

    char *buffer = re->argData;

    // Record the non-string arguments
    	NanoLogInternal::Log::recordPrimitive(buffer, arg0);
	NanoLogInternal::Log::recordPrimitive(buffer, arg1);
	NanoLogInternal::Log::recordPrimitive(buffer, arg2);


    // Record the strings (if any) at the end of the entry
    memcpy(buffer, arg3, str3Len); buffer += str3Len;*(reinterpret_cast<std::remove_const<typename std::remove_pointer<decltype(arg3)>::type>::type*>(buffer) - 1) = L'\0';

    // Make the entry visible
    NanoLogInternal::RuntimeLogger::finishAlloc(allocSize);
}


inline ssize_t
compressArgs__I32have32a32couple32of32things3237d443237f443237u443237s__testHelper47client46cc__31__(NanoLogInternal::Log::UncompressedEntry *re, char* out) {
    char *originalOutPtr = out;

    // Allocate nibbles
    BufferUtils::TwoNibbles *nib = reinterpret_cast<BufferUtils::TwoNibbles*>(out);
    out += 2;

    char *args = re->argData;

    // Read back all the primitives
    	int arg0; std::memcpy(&arg0, args, sizeof(int)); args +=sizeof(int);
	double arg1; std::memcpy(&arg1, args, sizeof(double)); args +=sizeof(double);
	unsigned int arg2; std::memcpy(&arg2, args, sizeof(unsigned int)); args +=sizeof(unsigned int);


    // Pack all the primitives
    	nib[0].first = 0x0f & static_cast<uint8_t>(BufferUtils::pack(&out, arg0));
	nib[0].second = 0x0f & static_cast<uint8_t>(BufferUtils::pack(&out, arg1));
	nib[1].first = 0x0f & static_cast<uint8_t>(BufferUtils::pack(&out, arg2));


    if (true) {
        // memcpy all the strings without compression
        size_t stringBytes = re->entrySize - (sizeof(arg0) + sizeof(arg1) + sizeof(arg2) +  0)
                                            - sizeof(NanoLogInternal::Log::UncompressedEntry);
        if (stringBytes > 0) {
            memcpy(out, args, stringBytes);
            out += stringBytes;
        }
    }

    return out - originalOutPtr;
}


inline void
decompressPrintArgs__I32have32a32couple32of32things3237d443237f443237u443237s__testHelper47client46cc__31__ (const char **in,
                        FILE *outputFd,
                        void (*aggFn)(const char*, ...)) {
    BufferUtils::TwoNibbles nib[2];
    memcpy(&nib, (*in), 2);
    (*in) += 2;

    // Unpack all the non-string argments
    	int arg0 = BufferUtils::unpack<int>(in, nib[0].first);
	double arg1 = BufferUtils::unpack<double>(in, nib[0].second);
	unsigned int arg2 = BufferUtils::unpack<unsigned int>(in, nib[1].first);


    // Find all the strings
    
                const char* arg3 = reinterpret_cast<const char*>(*in);
                (*in) += (strlen(arg3) + 1)*sizeof(*arg3); // +1 for null terminator
            

    const char *fmtString = "I have a couple of things %d, %f, %u, %s";
    const char *filename = "testHelper/client.cc";
    const int linenum = 31;
    const NanoLog::LogLevel logLevel = NOTICE;

    if (outputFd)
        fprintf(outputFd, "I have a couple of things %d, %f, %u, %s" "\r\n" , arg0, arg1, arg2, arg3);

    if (aggFn)
        (*aggFn)("I have a couple of things %d, %f, %u, %s" , arg0, arg1, arg2, arg3);
}


inline void __syang0__fl__I32have32a32double3237lf__testHelper47client46cc__30__(NanoLog::LogLevel level, const char* fmtStr , double arg0) {
    extern const uint32_t __fmtId__I32have32a32double3237lf__testHelper47client46cc__30__;

    if (level > NanoLog::getLogLevel())
        return;

    uint64_t timestamp = PerfUtils::Cycles::rdtsc();
    ;
    size_t allocSize = sizeof(arg0) +   sizeof(NanoLogInternal::Log::UncompressedEntry);
    NanoLogInternal::Log::UncompressedEntry *re = reinterpret_cast<NanoLogInternal::Log::UncompressedEntry*>(NanoLogInternal::RuntimeLogger::reserveAlloc(allocSize));

    re->fmtId = __fmtId__I32have32a32double3237lf__testHelper47client46cc__30__;
    re->timestamp = timestamp;
    re->entrySize = static_cast<uint32_t>(allocSize);

    char *buffer = re->argData;

    // Record the non-string arguments
    	NanoLogInternal::Log::recordPrimitive(buffer, arg0);


    // Record the strings (if any) at the end of the entry
    

    // Make the entry visible
    NanoLogInternal::RuntimeLogger::finishAlloc(allocSize);
}


inline ssize_t
compressArgs__I32have32a32double3237lf__testHelper47client46cc__30__(NanoLogInternal::Log::UncompressedEntry *re, char* out) {
    char *originalOutPtr = out;

    // Allocate nibbles
    BufferUtils::TwoNibbles *nib = reinterpret_cast<BufferUtils::TwoNibbles*>(out);
    out += 1;

    char *args = re->argData;

    // Read back all the primitives
    	double arg0; std::memcpy(&arg0, args, sizeof(double)); args +=sizeof(double);


    // Pack all the primitives
    	nib[0].first = 0x0f & static_cast<uint8_t>(BufferUtils::pack(&out, arg0));


    if (false) {
        // memcpy all the strings without compression
        size_t stringBytes = re->entrySize - (sizeof(arg0) +  0)
                                            - sizeof(NanoLogInternal::Log::UncompressedEntry);
        if (stringBytes > 0) {
            memcpy(out, args, stringBytes);
            out += stringBytes;
        }
    }

    return out - originalOutPtr;
}


inline void
decompressPrintArgs__I32have32a32double3237lf__testHelper47client46cc__30__ (const char **in,
                        FILE *outputFd,
                        void (*aggFn)(const char*, ...)) {
    BufferUtils::TwoNibbles nib[1];
    memcpy(&nib, (*in), 1);
    (*in) += 1;

    // Unpack all the non-string argments
    	double arg0 = BufferUtils::unpack<double>(in, nib[0].first);


    // Find all the strings
    

    const char *fmtString = "I have a double %lf";
    const char *filename = "testHelper/client.cc";
    const int linenum = 30;
    const NanoLog::LogLevel logLevel = NOTICE;

    if (outputFd)
        fprintf(outputFd, "I have a double %lf" "\r\n" , arg0);

    if (aggFn)
        (*aggFn)("I have a double %lf" , arg0);
}


inline void __syang0__fl__I32have32a32uint6495t3237lu__testHelper47client46cc__29__(NanoLog::LogLevel level, const char* fmtStr , unsigned long int arg0) {
    extern const uint32_t __fmtId__I32have32a32uint6495t3237lu__testHelper47client46cc__29__;

    if (level > NanoLog::getLogLevel())
        return;

    uint64_t timestamp = PerfUtils::Cycles::rdtsc();
    ;
    size_t allocSize = sizeof(arg0) +   sizeof(NanoLogInternal::Log::UncompressedEntry);
    NanoLogInternal::Log::UncompressedEntry *re = reinterpret_cast<NanoLogInternal::Log::UncompressedEntry*>(NanoLogInternal::RuntimeLogger::reserveAlloc(allocSize));

    re->fmtId = __fmtId__I32have32a32uint6495t3237lu__testHelper47client46cc__29__;
    re->timestamp = timestamp;
    re->entrySize = static_cast<uint32_t>(allocSize);

    char *buffer = re->argData;

    // Record the non-string arguments
    	NanoLogInternal::Log::recordPrimitive(buffer, arg0);


    // Record the strings (if any) at the end of the entry
    

    // Make the entry visible
    NanoLogInternal::RuntimeLogger::finishAlloc(allocSize);
}


inline ssize_t
compressArgs__I32have32a32uint6495t3237lu__testHelper47client46cc__29__(NanoLogInternal::Log::UncompressedEntry *re, char* out) {
    char *originalOutPtr = out;

    // Allocate nibbles
    BufferUtils::TwoNibbles *nib = reinterpret_cast<BufferUtils::TwoNibbles*>(out);
    out += 1;

    char *args = re->argData;

    // Read back all the primitives
    	unsigned long int arg0; std::memcpy(&arg0, args, sizeof(unsigned long int)); args +=sizeof(unsigned long int);


    // Pack all the primitives
    	nib[0].first = 0x0f & static_cast<uint8_t>(BufferUtils::pack(&out, arg0));


    if (false) {
        // memcpy all the strings without compression
        size_t stringBytes = re->entrySize - (sizeof(arg0) +  0)
                                            - sizeof(NanoLogInternal::Log::UncompressedEntry);
        if (stringBytes > 0) {
            memcpy(out, args, stringBytes);
            out += stringBytes;
        }
    }

    return out - originalOutPtr;
}


inline void
decompressPrintArgs__I32have32a32uint6495t3237lu__testHelper47client46cc__29__ (const char **in,
                        FILE *outputFd,
                        void (*aggFn)(const char*, ...)) {
    BufferUtils::TwoNibbles nib[1];
    memcpy(&nib, (*in), 1);
    (*in) += 1;

    // Unpack all the non-string argments
    	unsigned long int arg0 = BufferUtils::unpack<unsigned long int>(in, nib[0].first);


    // Find all the strings
    

    const char *fmtString = "I have a uint64_t %lu";
    const char *filename = "testHelper/client.cc";
    const int linenum = 29;
    const NanoLog::LogLevel logLevel = NOTICE;

    if (outputFd)
        fprintf(outputFd, "I have a uint64_t %lu" "\r\n" , arg0);

    if (aggFn)
        (*aggFn)("I have a uint64_t %lu" , arg0);
}


inline void __syang0__fl__I32have32an32integer3237d__testHelper47client46cc__28__(NanoLog::LogLevel level, const char* fmtStr , int arg0) {
    extern const uint32_t __fmtId__I32have32an32integer3237d__testHelper47client46cc__28__;

    if (level > NanoLog::getLogLevel())
        return;

    uint64_t timestamp = PerfUtils::Cycles::rdtsc();
    ;
    size_t allocSize = sizeof(arg0) +   sizeof(NanoLogInternal::Log::UncompressedEntry);
    NanoLogInternal::Log::UncompressedEntry *re = reinterpret_cast<NanoLogInternal::Log::UncompressedEntry*>(NanoLogInternal::RuntimeLogger::reserveAlloc(allocSize));

    re->fmtId = __fmtId__I32have32an32integer3237d__testHelper47client46cc__28__;
    re->timestamp = timestamp;
    re->entrySize = static_cast<uint32_t>(allocSize);

    char *buffer = re->argData;

    // Record the non-string arguments
    	NanoLogInternal::Log::recordPrimitive(buffer, arg0);


    // Record the strings (if any) at the end of the entry
    

    // Make the entry visible
    NanoLogInternal::RuntimeLogger::finishAlloc(allocSize);
}


inline ssize_t
compressArgs__I32have32an32integer3237d__testHelper47client46cc__28__(NanoLogInternal::Log::UncompressedEntry *re, char* out) {
    char *originalOutPtr = out;

    // Allocate nibbles
    BufferUtils::TwoNibbles *nib = reinterpret_cast<BufferUtils::TwoNibbles*>(out);
    out += 1;

    char *args = re->argData;

    // Read back all the primitives
    	int arg0; std::memcpy(&arg0, args, sizeof(int)); args +=sizeof(int);


    // Pack all the primitives
    	nib[0].first = 0x0f & static_cast<uint8_t>(BufferUtils::pack(&out, arg0));


    if (false) {
        // memcpy all the strings without compression
        size_t stringBytes = re->entrySize - (sizeof(arg0) +  0)
                                            - sizeof(NanoLogInternal::Log::UncompressedEntry);
        if (stringBytes > 0) {
            memcpy(out, args, stringBytes);
            out += stringBytes;
        }
    }

    return out - originalOutPtr;
}


inline void
decompressPrintArgs__I32have32an32integer3237d__testHelper47client46cc__28__ (const char **in,
                        FILE *outputFd,
                        void (*aggFn)(const char*, ...)) {
    BufferUtils::TwoNibbles nib[1];
    memcpy(&nib, (*in), 1);
    (*in) += 1;

    // Unpack all the non-string argments
    	int arg0 = BufferUtils::unpack<int>(in, nib[0].first);


    // Find all the strings
    

    const char *fmtString = "I have an integer %d";
    const char *filename = "testHelper/client.cc";
    const int linenum = 28;
    const NanoLog::LogLevel logLevel = NOTICE;

    if (outputFd)
        fprintf(outputFd, "I have an integer %d" "\r\n" , arg0);

    if (aggFn)
        (*aggFn)("I have an integer %d" , arg0);
}


inline void __syang0__fl__Notice32Level__testHelper47client46cc__24__(NanoLog::LogLevel level, const char* fmtStr ) {
    extern const uint32_t __fmtId__Notice32Level__testHelper47client46cc__24__;

    if (level > NanoLog::getLogLevel())
        return;

    uint64_t timestamp = PerfUtils::Cycles::rdtsc();
    ;
    size_t allocSize =   sizeof(NanoLogInternal::Log::UncompressedEntry);
    NanoLogInternal::Log::UncompressedEntry *re = reinterpret_cast<NanoLogInternal::Log::UncompressedEntry*>(NanoLogInternal::RuntimeLogger::reserveAlloc(allocSize));

    re->fmtId = __fmtId__Notice32Level__testHelper47client46cc__24__;
    re->timestamp = timestamp;
    re->entrySize = static_cast<uint32_t>(allocSize);

    char *buffer = re->argData;

    // Record the non-string arguments
    

    // Record the strings (if any) at the end of the entry
    

    // Make the entry visible
    NanoLogInternal::RuntimeLogger::finishAlloc(allocSize);
}


inline ssize_t
compressArgs__Notice32Level__testHelper47client46cc__24__(NanoLogInternal::Log::UncompressedEntry *re, char* out) {
    char *originalOutPtr = out;

    // Allocate nibbles
    BufferUtils::TwoNibbles *nib = reinterpret_cast<BufferUtils::TwoNibbles*>(out);
    out += 0;

    char *args = re->argData;

    // Read back all the primitives
    

    // Pack all the primitives
    

    if (false) {
        // memcpy all the strings without compression
        size_t stringBytes = re->entrySize - ( 0)
                                            - sizeof(NanoLogInternal::Log::UncompressedEntry);
        if (stringBytes > 0) {
            memcpy(out, args, stringBytes);
            out += stringBytes;
        }
    }

    return out - originalOutPtr;
}


inline void
decompressPrintArgs__Notice32Level__testHelper47client46cc__24__ (const char **in,
                        FILE *outputFd,
                        void (*aggFn)(const char*, ...)) {
    BufferUtils::TwoNibbles nib[0];
    memcpy(&nib, (*in), 0);
    (*in) += 0;

    // Unpack all the non-string argments
    

    // Find all the strings
    

    const char *fmtString = "Notice Level";
    const char *filename = "testHelper/client.cc";
    const int linenum = 24;
    const NanoLog::LogLevel logLevel = NOTICE;

    if (outputFd)
        fprintf(outputFd, "Notice Level" "\r\n" );

    if (aggFn)
        (*aggFn)("Notice Level" );
}


inline void __syang0__fl__Simple32log32message32with32032parameters__testHelper47client46cc__20__(NanoLog::LogLevel level, const char* fmtStr ) {
    extern const uint32_t __fmtId__Simple32log32message32with32032parameters__testHelper47client46cc__20__;

    if (level > NanoLog::getLogLevel())
        return;

    uint64_t timestamp = PerfUtils::Cycles::rdtsc();
    ;
    size_t allocSize =   sizeof(NanoLogInternal::Log::UncompressedEntry);
    NanoLogInternal::Log::UncompressedEntry *re = reinterpret_cast<NanoLogInternal::Log::UncompressedEntry*>(NanoLogInternal::RuntimeLogger::reserveAlloc(allocSize));

    re->fmtId = __fmtId__Simple32log32message32with32032parameters__testHelper47client46cc__20__;
    re->timestamp = timestamp;
    re->entrySize = static_cast<uint32_t>(allocSize);

    char *buffer = re->argData;

    // Record the non-string arguments
    

    // Record the strings (if any) at the end of the entry
    

    // Make the entry visible
    NanoLogInternal::RuntimeLogger::finishAlloc(allocSize);
}


inline ssize_t
compressArgs__Simple32log32message32with32032parameters__testHelper47client46cc__20__(NanoLogInternal::Log::UncompressedEntry *re, char* out) {
    char *originalOutPtr = out;

    // Allocate nibbles
    BufferUtils::TwoNibbles *nib = reinterpret_cast<BufferUtils::TwoNibbles*>(out);
    out += 0;

    char *args = re->argData;

    // Read back all the primitives
    

    // Pack all the primitives
    

    if (false) {
        // memcpy all the strings without compression
        size_t stringBytes = re->entrySize - ( 0)
                                            - sizeof(NanoLogInternal::Log::UncompressedEntry);
        if (stringBytes > 0) {
            memcpy(out, args, stringBytes);
            out += stringBytes;
        }
    }

    return out - originalOutPtr;
}


inline void
decompressPrintArgs__Simple32log32message32with32032parameters__testHelper47client46cc__20__ (const char **in,
                        FILE *outputFd,
                        void (*aggFn)(const char*, ...)) {
    BufferUtils::TwoNibbles nib[0];
    memcpy(&nib, (*in), 0);
    (*in) += 0;

    // Unpack all the non-string argments
    

    // Find all the strings
    

    const char *fmtString = "Simple log message with 0 parameters";
    const char *filename = "testHelper/client.cc";
    const int linenum = 20;
    const NanoLog::LogLevel logLevel = NOTICE;

    if (outputFd)
        fprintf(outputFd, "Simple log message with 0 parameters" "\r\n" );

    if (aggFn)
        (*aggFn)("Simple log message with 0 parameters" );
}


inline void __syang0__fl__This32is32a32string3237s__testHelper47client46cc__21__(NanoLog::LogLevel level, const char* fmtStr , const char* arg0) {
    extern const uint32_t __fmtId__This32is32a32string3237s__testHelper47client46cc__21__;

    if (level > NanoLog::getLogLevel())
        return;

    uint64_t timestamp = PerfUtils::Cycles::rdtsc();
    size_t str0Len = 1 + strlen(arg0);;
    size_t allocSize =  str0Len +  sizeof(NanoLogInternal::Log::UncompressedEntry);
    NanoLogInternal::Log::UncompressedEntry *re = reinterpret_cast<NanoLogInternal::Log::UncompressedEntry*>(NanoLogInternal::RuntimeLogger::reserveAlloc(allocSize));

    re->fmtId = __fmtId__This32is32a32string3237s__testHelper47client46cc__21__;
    re->timestamp = timestamp;
    re->entrySize = static_cast<uint32_t>(allocSize);

    char *buffer = re->argData;

    // Record the non-string arguments
    

    // Record the strings (if any) at the end of the entry
    memcpy(buffer, arg0, str0Len); buffer += str0Len;*(reinterpret_cast<std::remove_const<typename std::remove_pointer<decltype(arg0)>::type>::type*>(buffer) - 1) = L'\0';

    // Make the entry visible
    NanoLogInternal::RuntimeLogger::finishAlloc(allocSize);
}


inline ssize_t
compressArgs__This32is32a32string3237s__testHelper47client46cc__21__(NanoLogInternal::Log::UncompressedEntry *re, char* out) {
    char *originalOutPtr = out;

    // Allocate nibbles
    BufferUtils::TwoNibbles *nib = reinterpret_cast<BufferUtils::TwoNibbles*>(out);
    out += 0;

    char *args = re->argData;

    // Read back all the primitives
    

    // Pack all the primitives
    

    if (true) {
        // memcpy all the strings without compression
        size_t stringBytes = re->entrySize - ( 0)
                                            - sizeof(NanoLogInternal::Log::UncompressedEntry);
        if (stringBytes > 0) {
            memcpy(out, args, stringBytes);
            out += stringBytes;
        }
    }

    return out - originalOutPtr;
}


inline void
decompressPrintArgs__This32is32a32string3237s__testHelper47client46cc__21__ (const char **in,
                        FILE *outputFd,
                        void (*aggFn)(const char*, ...)) {
    BufferUtils::TwoNibbles nib[0];
    memcpy(&nib, (*in), 0);
    (*in) += 0;

    // Unpack all the non-string argments
    

    // Find all the strings
    
                const char* arg0 = reinterpret_cast<const char*>(*in);
                (*in) += (strlen(arg0) + 1)*sizeof(*arg0); // +1 for null terminator
            

    const char *fmtString = "This is a string %s";
    const char *filename = "testHelper/client.cc";
    const int linenum = 21;
    const NanoLog::LogLevel logLevel = NOTICE;

    if (outputFd)
        fprintf(outputFd, "This is a string %s" "\r\n" , arg0);

    if (aggFn)
        (*aggFn)("This is a string %s" , arg0);
}


inline void __syang0__fl__Warning32Level__testHelper47client46cc__25__(NanoLog::LogLevel level, const char* fmtStr ) {
    extern const uint32_t __fmtId__Warning32Level__testHelper47client46cc__25__;

    if (level > NanoLog::getLogLevel())
        return;

    uint64_t timestamp = PerfUtils::Cycles::rdtsc();
    ;
    size_t allocSize =   sizeof(NanoLogInternal::Log::UncompressedEntry);
    NanoLogInternal::Log::UncompressedEntry *re = reinterpret_cast<NanoLogInternal::Log::UncompressedEntry*>(NanoLogInternal::RuntimeLogger::reserveAlloc(allocSize));

    re->fmtId = __fmtId__Warning32Level__testHelper47client46cc__25__;
    re->timestamp = timestamp;
    re->entrySize = static_cast<uint32_t>(allocSize);

    char *buffer = re->argData;

    // Record the non-string arguments
    

    // Record the strings (if any) at the end of the entry
    

    // Make the entry visible
    NanoLogInternal::RuntimeLogger::finishAlloc(allocSize);
}


inline ssize_t
compressArgs__Warning32Level__testHelper47client46cc__25__(NanoLogInternal::Log::UncompressedEntry *re, char* out) {
    char *originalOutPtr = out;

    // Allocate nibbles
    BufferUtils::TwoNibbles *nib = reinterpret_cast<BufferUtils::TwoNibbles*>(out);
    out += 0;

    char *args = re->argData;

    // Read back all the primitives
    

    // Pack all the primitives
    

    if (false) {
        // memcpy all the strings without compression
        size_t stringBytes = re->entrySize - ( 0)
                                            - sizeof(NanoLogInternal::Log::UncompressedEntry);
        if (stringBytes > 0) {
            memcpy(out, args, stringBytes);
            out += stringBytes;
        }
    }

    return out - originalOutPtr;
}


inline void
decompressPrintArgs__Warning32Level__testHelper47client46cc__25__ (const char **in,
                        FILE *outputFd,
                        void (*aggFn)(const char*, ...)) {
    BufferUtils::TwoNibbles nib[0];
    memcpy(&nib, (*in), 0);
    (*in) += 0;

    // Unpack all the non-string argments
    

    // Find all the strings
    

    const char *fmtString = "Warning Level";
    const char *filename = "testHelper/client.cc";
    const int linenum = 25;
    const NanoLog::LogLevel logLevel = WARNING;

    if (outputFd)
        fprintf(outputFd, "Warning Level" "\r\n" );

    if (aggFn)
        (*aggFn)("Warning Level" );
}


} // end empty namespace

// Assignment of numerical ids to format NANO_LOG occurrences
extern const int __fmtId__Debug32level__testHelper47client46cc__23__ = 0; // testHelper/client.cc:23 "Debug level"
extern const int __fmtId__Error32Level__testHelper47client46cc__26__ = 1; // testHelper/client.cc:26 "Error Level"
extern const int __fmtId__I32have32a32couple32of32things3237d443237f443237u443237s__testHelper47client46cc__31__ = 2; // testHelper/client.cc:31 "I have a couple of things %d, %f, %u, %s"
extern const int __fmtId__I32have32a32double3237lf__testHelper47client46cc__30__ = 3; // testHelper/client.cc:30 "I have a double %lf"
extern const int __fmtId__I32have32a32uint6495t3237lu__testHelper47client46cc__29__ = 4; // testHelper/client.cc:29 "I have a uint64_t %lu"
extern const int __fmtId__I32have32an32integer3237d__testHelper47client46cc__28__ = 5; // testHelper/client.cc:28 "I have an integer %d"
extern const int __fmtId__Notice32Level__testHelper47client46cc__24__ = 6; // testHelper/client.cc:24 "Notice Level"
extern const int __fmtId__Simple32log32message32with32032parameters__testHelper47client46cc__20__ = 7; // testHelper/client.cc:20 "Simple log message with 0 parameters"
extern const int __fmtId__This32is32a32string3237s__testHelper47client46cc__21__ = 8; // testHelper/client.cc:21 "This is a string %s"
extern const int __fmtId__Warning32Level__testHelper47client46cc__25__ = 9; // testHelper/client.cc:25 "Warning Level"

// Start new namespace for generated ids and code
namespace GeneratedFunctions {

// Map of numerical ids to log message metadata
struct LogMetadata logId2Metadata[10] =
{
    {"Debug level", "testHelper/client.cc", 23, DEBUG},
{"Error Level", "testHelper/client.cc", 26, ERROR},
{"I have a couple of things %d, %f, %u, %s", "testHelper/client.cc", 31, NOTICE},
{"I have a double %lf", "testHelper/client.cc", 30, NOTICE},
{"I have a uint64_t %lu", "testHelper/client.cc", 29, NOTICE},
{"I have an integer %d", "testHelper/client.cc", 28, NOTICE},
{"Notice Level", "testHelper/client.cc", 24, NOTICE},
{"Simple log message with 0 parameters", "testHelper/client.cc", 20, NOTICE},
{"This is a string %s", "testHelper/client.cc", 21, NOTICE},
{"Warning Level", "testHelper/client.cc", 25, WARNING}
};

// Map of numerical ids to compression functions
ssize_t
(*compressFnArray[10]) (NanoLogInternal::Log::UncompressedEntry *re, char* out)
{
    compressArgs__Debug32level__testHelper47client46cc__23__,
compressArgs__Error32Level__testHelper47client46cc__26__,
compressArgs__I32have32a32couple32of32things3237d443237f443237u443237s__testHelper47client46cc__31__,
compressArgs__I32have32a32double3237lf__testHelper47client46cc__30__,
compressArgs__I32have32a32uint6495t3237lu__testHelper47client46cc__29__,
compressArgs__I32have32an32integer3237d__testHelper47client46cc__28__,
compressArgs__Notice32Level__testHelper47client46cc__24__,
compressArgs__Simple32log32message32with32032parameters__testHelper47client46cc__20__,
compressArgs__This32is32a32string3237s__testHelper47client46cc__21__,
compressArgs__Warning32Level__testHelper47client46cc__25__
};

// Map of numerical ids to decompression functions
void
(*decompressAndPrintFnArray[10]) (const char **in,
                                        FILE *outputFd,
                                        void (*aggFn)(const char*, ...))
{
    decompressPrintArgs__Debug32level__testHelper47client46cc__23__,
decompressPrintArgs__Error32Level__testHelper47client46cc__26__,
decompressPrintArgs__I32have32a32couple32of32things3237d443237f443237u443237s__testHelper47client46cc__31__,
decompressPrintArgs__I32have32a32double3237lf__testHelper47client46cc__30__,
decompressPrintArgs__I32have32a32uint6495t3237lu__testHelper47client46cc__29__,
decompressPrintArgs__I32have32an32integer3237d__testHelper47client46cc__28__,
decompressPrintArgs__Notice32Level__testHelper47client46cc__24__,
decompressPrintArgs__Simple32log32message32with32032parameters__testHelper47client46cc__20__,
decompressPrintArgs__This32is32a32string3237s__testHelper47client46cc__21__,
decompressPrintArgs__Warning32Level__testHelper47client46cc__25__
};

// Writes the metadata needed by the decompressor to interpret the log messages
// generated by compressFn.
long int writeDictionary(char *buffer, char *endOfBuffer) {
    using namespace NanoLogInternal::Log;
    char *startPos = buffer;
    
{
    // testHelper/client.cc:23 - "Debug level"
    FormatMetadata *fm;
    PrintFragment *pf;
    if (buffer + sizeof(FormatMetadata) + 21 >= endOfBuffer)
        return -1;

    fm = reinterpret_cast<FormatMetadata*>(buffer);
    buffer += sizeof(FormatMetadata);

    fm->numNibbles = 0;
    fm->numPrintFragments = 1;
    fm->logLevel = DEBUG;
    fm->lineNumber = 23;
    fm->filenameLength = 21;

    buffer = stpcpy(buffer, "testHelper/client.cc") + 1;

            // Fragment 0
            if (buffer + sizeof(PrintFragment)
                        + sizeof("Debug level")/sizeof(char) >= endOfBuffer)
                return -1;

            pf = reinterpret_cast<PrintFragment*>(buffer);
            buffer += sizeof(PrintFragment);

            pf->argType = NONE;
            pf->hasDynamicWidth = false;
            pf->hasDynamicPrecision = false;
            pf->fragmentLength = sizeof("Debug level")/sizeof(char);

            buffer = stpcpy(buffer, "Debug level") + 1;
}




{
    // testHelper/client.cc:26 - "Error Level"
    FormatMetadata *fm;
    PrintFragment *pf;
    if (buffer + sizeof(FormatMetadata) + 21 >= endOfBuffer)
        return -1;

    fm = reinterpret_cast<FormatMetadata*>(buffer);
    buffer += sizeof(FormatMetadata);

    fm->numNibbles = 0;
    fm->numPrintFragments = 1;
    fm->logLevel = ERROR;
    fm->lineNumber = 26;
    fm->filenameLength = 21;

    buffer = stpcpy(buffer, "testHelper/client.cc") + 1;

            // Fragment 0
            if (buffer + sizeof(PrintFragment)
                        + sizeof("Error Level")/sizeof(char) >= endOfBuffer)
                return -1;

            pf = reinterpret_cast<PrintFragment*>(buffer);
            buffer += sizeof(PrintFragment);

            pf->argType = NONE;
            pf->hasDynamicWidth = false;
            pf->hasDynamicPrecision = false;
            pf->fragmentLength = sizeof("Error Level")/sizeof(char);

            buffer = stpcpy(buffer, "Error Level") + 1;
}




{
    // testHelper/client.cc:31 - "I have a couple of things %d, %f, %u, %s"
    FormatMetadata *fm;
    PrintFragment *pf;
    if (buffer + sizeof(FormatMetadata) + 21 >= endOfBuffer)
        return -1;

    fm = reinterpret_cast<FormatMetadata*>(buffer);
    buffer += sizeof(FormatMetadata);

    fm->numNibbles = 3;
    fm->numPrintFragments = 4;
    fm->logLevel = NOTICE;
    fm->lineNumber = 31;
    fm->filenameLength = 21;

    buffer = stpcpy(buffer, "testHelper/client.cc") + 1;

            // Fragment 0
            if (buffer + sizeof(PrintFragment)
                        + sizeof("I have a couple of things %d")/sizeof(char) >= endOfBuffer)
                return -1;

            pf = reinterpret_cast<PrintFragment*>(buffer);
            buffer += sizeof(PrintFragment);

            pf->argType = int_t;
            pf->hasDynamicWidth = false;
            pf->hasDynamicPrecision = false;
            pf->fragmentLength = sizeof("I have a couple of things %d")/sizeof(char);

            buffer = stpcpy(buffer, "I have a couple of things %d") + 1;

            // Fragment 1
            if (buffer + sizeof(PrintFragment)
                        + sizeof(", %f")/sizeof(char) >= endOfBuffer)
                return -1;

            pf = reinterpret_cast<PrintFragment*>(buffer);
            buffer += sizeof(PrintFragment);

            pf->argType = double_t;
            pf->hasDynamicWidth = false;
            pf->hasDynamicPrecision = false;
            pf->fragmentLength = sizeof(", %f")/sizeof(char);

            buffer = stpcpy(buffer, ", %f") + 1;

            // Fragment 2
            if (buffer + sizeof(PrintFragment)
                        + sizeof(", %u")/sizeof(char) >= endOfBuffer)
                return -1;

            pf = reinterpret_cast<PrintFragment*>(buffer);
            buffer += sizeof(PrintFragment);

            pf->argType = unsigned_int_t;
            pf->hasDynamicWidth = false;
            pf->hasDynamicPrecision = false;
            pf->fragmentLength = sizeof(", %u")/sizeof(char);

            buffer = stpcpy(buffer, ", %u") + 1;

            // Fragment 3
            if (buffer + sizeof(PrintFragment)
                        + sizeof(", %s")/sizeof(char) >= endOfBuffer)
                return -1;

            pf = reinterpret_cast<PrintFragment*>(buffer);
            buffer += sizeof(PrintFragment);

            pf->argType = const_char_ptr_t;
            pf->hasDynamicWidth = false;
            pf->hasDynamicPrecision = false;
            pf->fragmentLength = sizeof(", %s")/sizeof(char);

            buffer = stpcpy(buffer, ", %s") + 1;
}




{
    // testHelper/client.cc:30 - "I have a double %lf"
    FormatMetadata *fm;
    PrintFragment *pf;
    if (buffer + sizeof(FormatMetadata) + 21 >= endOfBuffer)
        return -1;

    fm = reinterpret_cast<FormatMetadata*>(buffer);
    buffer += sizeof(FormatMetadata);

    fm->numNibbles = 1;
    fm->numPrintFragments = 1;
    fm->logLevel = NOTICE;
    fm->lineNumber = 30;
    fm->filenameLength = 21;

    buffer = stpcpy(buffer, "testHelper/client.cc") + 1;

            // Fragment 0
            if (buffer + sizeof(PrintFragment)
                        + sizeof("I have a double %lf")/sizeof(char) >= endOfBuffer)
                return -1;

            pf = reinterpret_cast<PrintFragment*>(buffer);
            buffer += sizeof(PrintFragment);

            pf->argType = double_t;
            pf->hasDynamicWidth = false;
            pf->hasDynamicPrecision = false;
            pf->fragmentLength = sizeof("I have a double %lf")/sizeof(char);

            buffer = stpcpy(buffer, "I have a double %lf") + 1;
}




{
    // testHelper/client.cc:29 - "I have a uint64_t %lu"
    FormatMetadata *fm;
    PrintFragment *pf;
    if (buffer + sizeof(FormatMetadata) + 21 >= endOfBuffer)
        return -1;

    fm = reinterpret_cast<FormatMetadata*>(buffer);
    buffer += sizeof(FormatMetadata);

    fm->numNibbles = 1;
    fm->numPrintFragments = 1;
    fm->logLevel = NOTICE;
    fm->lineNumber = 29;
    fm->filenameLength = 21;

    buffer = stpcpy(buffer, "testHelper/client.cc") + 1;

            // Fragment 0
            if (buffer + sizeof(PrintFragment)
                        + sizeof("I have a uint64_t %lu")/sizeof(char) >= endOfBuffer)
                return -1;

            pf = reinterpret_cast<PrintFragment*>(buffer);
            buffer += sizeof(PrintFragment);

            pf->argType = unsigned_long_int_t;
            pf->hasDynamicWidth = false;
            pf->hasDynamicPrecision = false;
            pf->fragmentLength = sizeof("I have a uint64_t %lu")/sizeof(char);

            buffer = stpcpy(buffer, "I have a uint64_t %lu") + 1;
}




{
    // testHelper/client.cc:28 - "I have an integer %d"
    FormatMetadata *fm;
    PrintFragment *pf;
    if (buffer + sizeof(FormatMetadata) + 21 >= endOfBuffer)
        return -1;

    fm = reinterpret_cast<FormatMetadata*>(buffer);
    buffer += sizeof(FormatMetadata);

    fm->numNibbles = 1;
    fm->numPrintFragments = 1;
    fm->logLevel = NOTICE;
    fm->lineNumber = 28;
    fm->filenameLength = 21;

    buffer = stpcpy(buffer, "testHelper/client.cc") + 1;

            // Fragment 0
            if (buffer + sizeof(PrintFragment)
                        + sizeof("I have an integer %d")/sizeof(char) >= endOfBuffer)
                return -1;

            pf = reinterpret_cast<PrintFragment*>(buffer);
            buffer += sizeof(PrintFragment);

            pf->argType = int_t;
            pf->hasDynamicWidth = false;
            pf->hasDynamicPrecision = false;
            pf->fragmentLength = sizeof("I have an integer %d")/sizeof(char);

            buffer = stpcpy(buffer, "I have an integer %d") + 1;
}




{
    // testHelper/client.cc:24 - "Notice Level"
    FormatMetadata *fm;
    PrintFragment *pf;
    if (buffer + sizeof(FormatMetadata) + 21 >= endOfBuffer)
        return -1;

    fm = reinterpret_cast<FormatMetadata*>(buffer);
    buffer += sizeof(FormatMetadata);

    fm->numNibbles = 0;
    fm->numPrintFragments = 1;
    fm->logLevel = NOTICE;
    fm->lineNumber = 24;
    fm->filenameLength = 21;

    buffer = stpcpy(buffer, "testHelper/client.cc") + 1;

            // Fragment 0
            if (buffer + sizeof(PrintFragment)
                        + sizeof("Notice Level")/sizeof(char) >= endOfBuffer)
                return -1;

            pf = reinterpret_cast<PrintFragment*>(buffer);
            buffer += sizeof(PrintFragment);

            pf->argType = NONE;
            pf->hasDynamicWidth = false;
            pf->hasDynamicPrecision = false;
            pf->fragmentLength = sizeof("Notice Level")/sizeof(char);

            buffer = stpcpy(buffer, "Notice Level") + 1;
}




{
    // testHelper/client.cc:20 - "Simple log message with 0 parameters"
    FormatMetadata *fm;
    PrintFragment *pf;
    if (buffer + sizeof(FormatMetadata) + 21 >= endOfBuffer)
        return -1;

    fm = reinterpret_cast<FormatMetadata*>(buffer);
    buffer += sizeof(FormatMetadata);

    fm->numNibbles = 0;
    fm->numPrintFragments = 1;
    fm->logLevel = NOTICE;
    fm->lineNumber = 20;
    fm->filenameLength = 21;

    buffer = stpcpy(buffer, "testHelper/client.cc") + 1;

            // Fragment 0
            if (buffer + sizeof(PrintFragment)
                        + sizeof("Simple log message with 0 parameters")/sizeof(char) >= endOfBuffer)
                return -1;

            pf = reinterpret_cast<PrintFragment*>(buffer);
            buffer += sizeof(PrintFragment);

            pf->argType = NONE;
            pf->hasDynamicWidth = false;
            pf->hasDynamicPrecision = false;
            pf->fragmentLength = sizeof("Simple log message with 0 parameters")/sizeof(char);

            buffer = stpcpy(buffer, "Simple log message with 0 parameters") + 1;
}




{
    // testHelper/client.cc:21 - "This is a string %s"
    FormatMetadata *fm;
    PrintFragment *pf;
    if (buffer + sizeof(FormatMetadata) + 21 >= endOfBuffer)
        return -1;

    fm = reinterpret_cast<FormatMetadata*>(buffer);
    buffer += sizeof(FormatMetadata);

    fm->numNibbles = 0;
    fm->numPrintFragments = 1;
    fm->logLevel = NOTICE;
    fm->lineNumber = 21;
    fm->filenameLength = 21;

    buffer = stpcpy(buffer, "testHelper/client.cc") + 1;

            // Fragment 0
            if (buffer + sizeof(PrintFragment)
                        + sizeof("This is a string %s")/sizeof(char) >= endOfBuffer)
                return -1;

            pf = reinterpret_cast<PrintFragment*>(buffer);
            buffer += sizeof(PrintFragment);

            pf->argType = const_char_ptr_t;
            pf->hasDynamicWidth = false;
            pf->hasDynamicPrecision = false;
            pf->fragmentLength = sizeof("This is a string %s")/sizeof(char);

            buffer = stpcpy(buffer, "This is a string %s") + 1;
}




{
    // testHelper/client.cc:25 - "Warning Level"
    FormatMetadata *fm;
    PrintFragment *pf;
    if (buffer + sizeof(FormatMetadata) + 21 >= endOfBuffer)
        return -1;

    fm = reinterpret_cast<FormatMetadata*>(buffer);
    buffer += sizeof(FormatMetadata);

    fm->numNibbles = 0;
    fm->numPrintFragments = 1;
    fm->logLevel = WARNING;
    fm->lineNumber = 25;
    fm->filenameLength = 21;

    buffer = stpcpy(buffer, "testHelper/client.cc") + 1;

            // Fragment 0
            if (buffer + sizeof(PrintFragment)
                        + sizeof("Warning Level")/sizeof(char) >= endOfBuffer)
                return -1;

            pf = reinterpret_cast<PrintFragment*>(buffer);
            buffer += sizeof(PrintFragment);

            pf->argType = NONE;
            pf->hasDynamicWidth = false;
            pf->hasDynamicPrecision = false;
            pf->fragmentLength = sizeof("Warning Level")/sizeof(char);

            buffer = stpcpy(buffer, "Warning Level") + 1;
}


    return buffer - startPos;
}

// Total number of logIds. Can be used to bounds check array accesses.
size_t numLogIds = 10;

// Pop the unused gcc warnings
#pragma GCC diagnostic pop

}; // GeneratedFunctions

#endif /* BUFFER_STUFFER */
