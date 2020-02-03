#pragma once

#define CUDA_CHECK( code )                                                     \
{                                                                              \
  cudaError_t err__ = code;                                                    \
  if( err__ != cudaSuccess )                                                   \
  {                                                                            \
    std::cerr << "Error in file " << __FILE__ <<" on line " << __LINE__ << ":" \
              << cudaGetErrorString( err__ ) << std::endl;                     \
    exit(1);                                                                   \
  }                                                                            \
}


enum BufferType 
{
    HOST_BUFFER,
    DEVICE_BUFFER,
};

template<typename T>
class BufferBase {
public:
    BufferBase(BufferType type, size_t count = 0) :
        m_type(type), m_ptr(0), m_count(count) 
    {
        alloc(count);
    }

    ~BufferBase()
    {
        free();
    }

    void alloc(size_t count)
    {
        alloc(count, m_type);
    }

    void alloc(size_t count, BufferType type) 
    {
        if (m_ptr) 
        {
            free();
        }

        m_count = count;
        m_type = type;

        if (count > 0) 
        {
            if (m_type == HOST_BUFFER) 
            {
                m_ptr = new T[m_count];
            }
            else 
            {                
                 CUDA_CHECK( cudaMalloc(&m_ptr, sizeInByte()) );
            }
        }

        clear();
    }

    void free() 
    {
        if (m_ptr) 
        {
            if (m_type == HOST_BUFFER) 
            {
                delete[] m_ptr;
            }
            else
            {                
                CUDA_CHECK( cudaFree(m_ptr) );                
            }
        }
        m_ptr = 0;
        m_count = 0;        
    }

    void resize(const size_t count) 
    {
        BufferBase buffer(m_type, count);
        buffer.copyFrom(min(count, m_count), m_type, m_ptr);

        std::swap(m_type,  buffer.m_type);
        std::swap(m_ptr,   buffer.m_ptr);
        std::swap(m_count, buffer.m_count);
    }

    void copyFrom(const size_t count, const BufferType src_type, const T* src_ptr) 
    {
        if (count == 0)
        {
            return;
        }

        if (count > m_count) 
        {
            alloc(count);
        }

        if (m_type == HOST_BUFFER)
        {
            if (src_type == HOST_BUFFER)
            {
                memcpy(m_ptr, src_ptr, sizeof(T) * count);
            }
            else
            {
                CUDA_CHECK( cudaMemcpy(m_ptr, src_ptr, sizeof(T) * count, cudaMemcpyDeviceToHost) );
            }
        }
        else
        {
            if (src_type == HOST_BUFFER)
            {
                CUDA_CHECK(cudaMemcpy(m_ptr, src_ptr, sizeof(T) * count, cudaMemcpyHostToDevice));
            }
            else
            {
                CUDA_CHECK(cudaMemcpy(m_ptr, src_ptr, sizeof(T) * count, cudaMemcpyDeviceToDevice));
            }
        }
    }

    void clear()
    {
        if (m_ptr)
        {
            if (m_type == HOST_BUFFER)
            {
                memset(m_ptr, 0, sizeInByte());
            }
            else
            {
                CUDA_CHECK( cudaMemset(m_ptr, 0, sizeInByte()) );
            }
        }
    }

    T operator[] (const size_t i) const
    {
        if (m_type == HOST_BUFFER)
        {
            return m_ptr[i];
        }
        else
        {
            T t;
            CUDA_CHECK( cudaMemcpy(&t, m_ptr + i, sizeof(T), cudaMemcpyDeviceToHost) );
            return t;
        }
    }

    T& operator[] (const size_t i)
    {
        if (m_type == HOST_BUFFER)
        {
            return m_ptr[i];
        }
        else
        {
            T t;
            CUDA_CHECK(cudaMemcpy(&t, m_ptr + i, sizeof(T), cudaMemcpyDeviceToHost));
            return t;
        }
    }
    
    size_t size() const { return m_count; }
    T* data() const { return m_ptr; }
    BufferType type() const { return m_type; }

protected:
    size_t sizeInByte() const { return m_count * sizeof(T); }

protected:
    BufferType m_type;
    T* m_ptr;
    size_t m_count;
};

template<BufferType TYPE, typename T>
class Buffer : public BufferBase<T> {
public:
    Buffer(const size_t count = 0) :
        BufferBase(TYPE, count) 
    {
    }

    template<BufferType UTYPE, typename T>
    Buffer(Buffer<UTYPE, T> buffer)
        : BufferBase(TYPE, buffer.size())
    {
        copyFrom(buffer.size(), buffer.type(), buffer.data());
    }

    template<BufferType COPY_TYPE>
    Buffer<TYPE, T>& operator = (const Buffer<COPY_TYPE, T>& buffer) 
    {
        copyFrom(buffer.m_count, buffer.m_type, buffer.m_ptr);
        return *this;
    }
};

//template class BufferBase<float3>;
//template class BufferBase<HOST_BUFFER, float3>;